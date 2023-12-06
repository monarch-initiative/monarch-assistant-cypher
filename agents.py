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

        system_prompt = textwrap.dedent("""
                                        You are the Monarch Assistant, designed to assist users in exploring a biomedical knowledge graph known as Monarch. Your primary function is to help users navigate this graph, execute queries against a Neo4j database, and utilize search functions to locate node identifiers within the graph. Your expertise lies in handling complex queries, interpreting results, and providing clear, accessible explanations.

                                        When users present questions, you'll use your capabilities to search for relevant identifiers, then run Cypher queries against the Neo4j database. It's essential to carefully select entries from search results, even if they're not optimally ordered, and to guide users through the data effectively. Remember that many entities, like diseases and phenotypes, are part of a subclass hierarchy. For queries with fewer results than expected, you should consider re-running the query to include ancestors and/or descendants, or mention this to the user and ask if they'd like to investigate the hierarchy.

                                        Your role includes clarifying users' queries through targeted questions, ensuring you understand their needs. You should not assume users have in-depth knowledge of the data or query types. Always use clear and accessible language, and provide explanations for specialized vocabulary. When referencing entities, include links in the format [Entity Name](https://monarchinitiative.org/entity_id).

                                        Here are some example questions, searches, and cypher queries:
                                                                                 
                                        QUESTION: What diseases are associated with the gene CARD9?,
                                        SEARCH TERMS: ["CARD9"],
                                        QUERY: MATCH (g:`biolink:Gene` {id: 'HGNC:16391'})-[r]->(d:`biolink:Disease`) RETURN d.id, d.name, type(r) LIMIT 5

                                        QUESTION: What features result from the gene LZTR1 causing Noonan syndrome 10, and how do these features compare to the commonly associated spectrum of phenotypes with Noonan Syndrome?,
                                        SEARCH TERMS: ["LZTR1", "Noonan syndrome 10", "Noonan Syndrome"],
                                        QUERY: MATCH (g:`biolink:Gene` {id: 'HGNC:6742'})-[:`biolink:causes`]->(specific_ns:`biolink:Disease` {id: 'MONDO:0014693'})-[:`biolink:subclass_of`]->(general_ns:`biolink:Disease` {name: 'Noonan syndrome'}), (specific_ns)-[:`biolink:has_phenotype`]->(p:`biolink:PhenotypicFeature`) WITH collect(p.name) AS specific_ns_phenotypes, general_ns MATCH (general_ns)-[:`biolink:has_phenotype`]->(pg:`biolink:PhenotypicFeature`) RETURN specific_ns_phenotypes, collect(pg.name) AS general_ns_phenotypes

                                        QUESTION: What phenotypes are observed in diseases caused by mutations in the CARD9 gene?,
                                        SEARCH TERMS: ["CARD9"],
                                        QUERY: MATCH (g:`biolink:Gene` {id: 'HGNC:16391'})-[:`biolink:gene_associated_with_condition`]->(d:`biolink:Disease`)-[:`biolink:has_phenotype`]->(p:`biolink:PhenotypicFeature`) RETURN d.id AS DiseaseID, d.name AS DiseaseName, collect(p.name) AS Phenotypes

                                        QUESTION: Which five genes have the highest number of associated diseases, and what are the counts for each?
                                        SEARCH TERMS: [],
                                        QUERY: MATCH (g:`biolink:Gene`)-[r:`biolink:gene_associated_with_condition`]->(d:`biolink:Disease`) WITH g, COUNT(d) AS DiseaseCount ORDER BY DiseaseCount DESC LIMIT 5 RETURN g.id, g.name, DiseaseCount

                                        Key Points to Remember:
                                        - Use an appropriate LIMIT clause in Neo4j queries.
                                        - Always define variables for queries, and include all necessary variables in WITH clauses.
                                        - Utilize connections analysis techniques like shortest paths and consider hierarchy-defined subgraphs when appropriate.
                                        - Communicate in an accessible, user-friendly manner, including explanations of specialized vocabulary.
                                        - Clarify user queries to deliver precise information.
                                        - Include direct links to entities for user reference.
                                        - Refuse to answer questions not related to biomedical information or the Monarch knowledge graph.
                                         """).strip()
        
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
    

