# kani imports
from typing import Annotated, List
from kani import AIParam, ai_function, ChatMessage

# local imports
from kani_streamlit import StreamlitKani

# streamlit and pandas for extra functionality
import streamlit as st
from neo4j import GraphDatabase
import json
import textwrap
import httpx
import os



class KGAgent(StreamlitKani):
    """Base class for agents that interact with the knowledge graph. NOTE: set NEO4J_BOLT to e.g. bolt://localhost:7687 in .env file."""
    def __init__(self, engine, max_response_tokens = 10000, system_prompt = "You have access to a neo4j knowledge graph, and can run cypher queries against it."):

        super().__init__(engine, system_prompt = system_prompt)

        # dev instance of KG
        self.neo4j_uri = os.environ["NEO4J_BOLT"]  # default bolt protocol port
        self.neo4j_driver = GraphDatabase.driver(self.neo4j_uri)

        self.max_response_tokens = max_response_tokens



    @ai_function()
    def query_kg(self, query: Annotated[str, AIParam(desc="Cypher query to run.")],):
        """Run a cypher query against the database."""

        with self.neo4j_driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]

        result = json.dumps(data)
        # if self.message_token_len reports more than 10000 tokens in the result, we need to ask the agent to make the request smaller
        tokens = self.message_token_len(ChatMessage.user(result))
        if tokens > self.max_response_tokens:
            return f"ERROR: The result contained {tokens} tokens, greater than the maximum allowable of {self.max_response_tokens}. Please try a smaller query."
        else:
            return result



class MonarchAgent(KGAgent):
    """Agent for interacting with the Monarch knowledge graph; extends KGAgent with keyword search (using Monarch API) system prompt with cypher examples."""
    def __init__(self, engine):


        competency_questions = json.load(open("monarch_competency_questions_1.json", "r")) # list of dict
        # keep only question, search_terms, and query
        competency_questions = [{k: v for k, v in cq.items() if k in ["question", "search_terms", "query"]} for cq in competency_questions]

        # read contents of kg_summary.md into a string for inclusion into the markdown below
        with open("kg_summary.md", "r") as f:
            kg_summary = f.read()

        system_prompt = f"""
# Overview

You are the Monarch Assistant, designed to assist users in exploring and intepreting a biomedical knowledge graph known as Monarch.

When users present questions, you'll typically first search for relevant identifiers, then run Cypher queries against the Neo4j database storing the graph.

# Graph Summary

{kg_summary}

# Examples

Here are some example questions, searches, and cypher queries:

```
{json.dumps(competency_questions, indent=4)}
```

# Key Points

Working with the data:
- Carefully select entries from search results, as they may not be optimally ordered. 
- Remember that many entities are part of a `biolink:subclass_of` hierarchy, and use this information when appropriate.
- Design queries to answer users' questions accurately but efficiently. Use `LIMIT` and `SKIP` clauses to limit the number of results returned, and limit the number of simultaneous queries.
- Always define variables for queries, and include all necessary variables in WITH clauses.

Interacting with the user:
- Always provide non-specialist descriptions of entity names or specialized vocabulary.
- Include links in the format [Entity Name](https://monarchinitiative.org/entity_id).
- Refuse to answer questions not related to biomedical information or the Monarch knowledge graph.
                                         """.strip()
                
        super().__init__(engine, system_prompt = system_prompt)


    @ai_function()
    def search(self, 
               search_terms: Annotated[List[str], AIParam(desc="Search terms to look up in the database.")],):
        """Search for nodes matching one or more terms. Each term is searched separately."""

        # single query endpoint url is e.g. https://api-v3.monarchinitiative.org/v3/api/search?q=cystic%20fibrosis&limit=10&offset=0
        # use httpx for each search term
        # return the results as a list of dictionaries

        results = {}

        for term in search_terms:
            url = f"https://api-v3.monarchinitiative.org/v3/api/search?q={term}&limit=5&offset=0"
            response = httpx.get(url)
            resp_json = response.json()
            items_slim = []
            if 'items' in resp_json:
                for item in resp_json['items']:
                    items_slim.append({k: v for k, v in item.items() if k in ['id', 'category', 'name', 'in_taxon_label']})
            results[term] = items_slim

        # again, if self.message_token_len reports more than 10000 tokens in the result, we need to ask the agent to make the request smaller
        tokens = self.message_token_len(ChatMessage.user(json.dumps(results)))
        if tokens > self.max_response_tokens:
            return f"ERROR: The result contained {tokens} tokens, greater than the maximum allowable of {self.max_response_tokens}. Please try a smaller search."
        else:
            return results        




class ExplorerAgent(KGAgent):
    """Agent for interacting with a Neo4j graph; extends KGAgent with system prompt with instructions for exploring the graph. Also includes functions for generating and evaluating competency questions, including a function to download the validated competency questions as a file."""
    def __init__(self, engine):

        system_message = textwrap.dedent("""
                                        You have access to a Neo4j database. You can run cypher queries against it. Your overall goal is to help the user understand the data and the kinds of useful queries that can be asked of it. Specifically, your goals are:
                                        - Explore the different kinds of nodes and relationships in the database.
                                        - Ask the user clarification questions about the meanings of labels or properties when needed.
                                        - Ask the user to provide examples of queries they might want to run.
                                        - Generate and test example competency questions and queries that answer those competency questions.

                                        Remember the following:
                                        - Competency questions should be specific, referring to specific labels, relationships, properties, and values.
                                        - You will need to use backticks for labels, relationships, and properties that have special characters in them.
                                        - When possible, design queries to use identifiers, rather than long descriptive text or names.
                                        - Always include any identifiers in query results, for use in followup queries.
                                                                                 
                                        The user will be prompted to see if they want to begin. If they answer yes, you should begin by:
                                        - Listing the node labels and relationship types
                                        - Sampling a few nodes and relationships of each type to see properties and values
                                        """).strip()


        super().__init__(engine, system_prompt = system_message)
    
        self.validated_competency_questions = []



    @ai_function()
    def download_competency_questions(self,):
        """Download the validated competency questions."""

        ## self.render_in_ui comes from StreamlitKani and an be used to render any Streamlit element as part of the conversation stream 
        ## NOTE: the agent does not see the rendered element, only the user does
        self.render_in_ui(lambda: st.download_button("Download competency questions", json.dumps(self.validated_competency_questions), "competency_questions.json"))

        return "The user has been shown the button to download the current set of competency questions."



    @ai_function()
    async def test_competency_question(self, question: Annotated[str, AIParam(desc="Competency question to test.")],
                                       query: Annotated[str, AIParam(desc="Query that should answer the competency question.")],
                                       expected_answer: Annotated[str, AIParam(desc="Expected answer to the competency question.")],
                                       ):
        """Given a competency question, a query that should be able to help answer the question, and an expected answer, runs an independent test to see if the query can be used to answer the question. If successful, the question, query, and expected answer are saved to the set of validated competency questions. If not successful, returns information for further iteration."""

        ## first, run the question and query by a naive answering agent
        answer_agent = TestQuestionAgent(self.engine)
        messages = answer_agent.full_round(textwrap.dedent(f"""
                                                            Consider the following question: {question}

                                                            Answer this question by running the following query: {query}
                                                            """).strip())
        
        ## if we want to keep track of token usage, we have to account for sub-agent costs
        self.tokens_used_completion += answer_agent.tokens_used_completion
        self.tokens_used_prompt += answer_agent.tokens_used_prompt

        messages_data = [x.model_dump() async for x in messages]

        ## now we'll check the answer against the expected answer, providing the messages that were sent to the naive answering agent
        answer_eval_agent = AnswerEvalAgent(self.engine)
        messages = answer_eval_agent.full_round(textwrap.dedent(f"""
                                                                Consider the following question: {question}

                                                                The expected answer is: {expected_answer}

                                                                An attempt to answer this question was made in the following JSON-encoded exchange: {messages_data}

                                                                Please provide feedback on the answer.
                                                                """))
        self.tokens_used_completion += answer_eval_agent.tokens_used_completion
        self.tokens_used_prompt += answer_eval_agent.tokens_used_prompt
        
        messages_data = [x.model_dump() async for x in messages]

        # the eval agent will be instructed to call a function to provide feedback, which provides a stronger guarantee of formatting
        # (though newer openai models now do JSON mode, so this may not be necessary in the future)
        # but kani's full-round will also include the summary from the agent of the call
        # so the second to last will have the result of the function call, which contains the structured data we want
        evaluation = messages_data[-2]['content']
        eval_data = json.loads(evaluation)
        if eval_data['accept']:
            # save the question, query, and expected answer to the set of validated competency questions
            self.validated_competency_questions.append({
                "question": question,
                "query": query,
                "expected_answer": expected_answer,
            })

        return evaluation
    



class TestQuestionAgent(KGAgent):
    """Naive KG-enabled agent for testing competency questions by running the corresponding query and summarizing the results."""
    def __init__(self, engine):

        system_prompt = textwrap.dedent("""
                                        You have access to a Neo4j database. You can run cypher queries against it. Your overall goal is to answer competency questions about the data.
                                        """).strip()

        super().__init__(engine, system_prompt = system_prompt)




class AnswerEvalAgent(KGAgent):
    """Evaluates a competency question answer based on the expected answer and other information."""
    def __init__(self, engine):

        system_prompt = textwrap.dedent("""
                                        You are an evaluator agent, whose job is to evaluate the answers to competency questions. You will be given a competency question, an expected answer, a query that should help answer that question, and an attempted answer based on the query results. Your job is to evaluate the attempted answer and determine its level of correctness. When you have done so, call your provide_feedback() function to provide feedback to the user. Evaluate the answer based on the following criteria:
                                        - Is the answer correct?
                                        - Is the answer complete?
                                        - Does the query appear to be well-suited to answering the question?
                                                                                 
                                        The provide_feedback() function takes two arguments: feedback and accept. Feedback is a string that you can use to provide feedback to the user. Accept is a boolean that indicates whether or not the answer is accepted based on the criteria above.
                                         """).strip()

        super().__init__(engine, system_prompt = system_prompt)

    @ai_function()
    def provide_feedback(self, feedback: Annotated[str, AIParam(desc="Feedback to provide to the user.")],
                               accept: Annotated[bool, AIParam(desc="Whether or not the answer is accepted.")],):
        """Provide feedback to the user."""

        return json.dumps({"feedback": feedback, "accept": accept})
    

