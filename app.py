# Example usage of StreamlitKani

########################
##### 0 - load libs
########################

# for reading API keys from .env file
import os
import dotenv # pip install python-dotenv
import dotenv
dotenv.load_dotenv() 

# kani related imports
import kani_streamlit as ks
from kani.engines.openai import OpenAIEngine
from agents import ExplorerAgent, MonarchAgent
import textwrap

########################
##### 1 - Configuration
########################

# read API keys .env file (e.g. set OPENAI_API_KEY=.... in .env and gitignore .env)
import dotenv
dotenv.load_dotenv() 


# initialize the application and set some page settings
# parameters here are passed to streamlit.set_page_config, 
# see more at https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
# this function MUST be run first
ks.initialize_app_config(
    page_title = "Cypher Explore",
    page_icon = "፨", # can also be a URL
    initial_sidebar_state = "expanded", # or "expanded"
    menu_items = {
            "Get Help": "https://github.com/.../issues",
            "Report a Bug": "https://github.com/.../issues",
            "About": "An agent for exploring neo4j graphs.",
        }
)


########################
##### 2 - Define Agents
########################

# StreamlitKani agents are Kani agents and work the same
# We must subclass StreamlitKani instead of Kani to get the Streamlit UI
# define an engine to use (see Kani documentation for more info)
engine = OpenAIEngine(os.environ["OPENAI_API_KEY"], model="gpt-4-1106-preview")


# We also have to define a function that returns a dictionary of agents to serve
# Agents are keyed by their name, which is what the user will see in the UI
def get_agents():
    return {
        "Monarch Assistant (2.0)": {
            "agent": MonarchAgent(engine),
            # The greeting is not seen by the agent, but is shown to the user to provide instructions
            "greeting": textwrap.dedent(f"""
                                        I'm the Monarch Assistant, an AI chatbot with access to the [Monarch Inititive](https://monarchinitiative.org) biomedical knowledgebase. I can search for information on diseases, genes, and phenotypes. Here are some things you might try asking:

                                        - What is the genetic basis of Cystic Fibrosis?
                                        - What symptoms are associated with Fanconi Anemia?
                                        - Which symptoms are shared by two or more forms of Progressive External Ophthalmoplegia?
                                        
                                        Please note that I am a research preview, and this information should not be used for diagnoses, clinical decision making, or other medical applications.
                                        """),
            "description": "An assistant for exploring the Monarch Iniative biomedical knowledge graph.",
            "avatar": "https://avatars.githubusercontent.com/u/5161984?s=200&v=4", # these can also be URLs
            "user_avatar": "👤",
            "token_costs": {"prompt": 0.01, "completion": 0.03}
        },
        "Competency Question Agent": {
            "agent": ExplorerAgent(engine),
            # The greeting is not seen by the agent, but is shown to the user to provide instructions
            "greeting": "Hello! I am here to help you explore a neo4j graph, understand the kinds of questions it can answer and queries to answer those questions, and test those queries to develop a set of competency questions. We'll start by looking at the different kinds of nodes and relationships in the graph. Should we begin?",
            "description": "An agent for exploring Neo4j graphs, and generating and evaluating competency questions over them.",
            "avatar": "👾", # these can also be URLs
            "user_avatar": "👤",
            "token_costs": {"prompt": 0.01, "completion": 0.03}
        },

    }


# tell the app to use that function to create agents when needed
ks.set_app_agents(get_agents)


########################
##### 3 - Serve App
########################

ks.serve_app()
