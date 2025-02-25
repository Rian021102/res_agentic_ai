from phi.agent import Agent, RunResponse
from phi.utils.pprint import pprint_run_response
from phi.tools.python import PythonTools
from phi.model.openai import OpenAIChat
from langchain_openai import OpenAIEmbeddings
from phi.knowledge.langchain import LangChainKnowledgeBase
import os
from langchain_community.vectorstores import DeepLake
import streamlit as st

api_key = os.getenv("OPENAI_API_KEY")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")

# Validate environment variables
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
if not ACTIVELOOP_TOKEN:
    raise ValueError("ACTIVELOOP_TOKEN environment variable is not set")

# # Initialize OpenAI embeddings with error handling
try:
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
except Exception as e:
    raise Exception(f"Failed to initialize OpenAI embeddings: {str(e)}")

# Authenticate with ActiveLoop
os.environ['ACTIVELOOP_TOKEN'] = ACTIVELOOP_TOKEN

# Global variable to store the database instance
def get_db_instance():
    if 'db_instance' not in st.session_state:
        st.session_state.db_instance = DeepLake(dataset_path="hub://rian/reseng", embedding=embeddings_model, read_only=True)
    return st.session_state.db_instance

db = get_db_instance()
retriever = db.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever,vectorstore=db)

knowledge_agent = Agent(
    name="RAG Agent",
    role="Retrieve information from the document in knowledge base",
    instructions=["Retrieve information based on query from vector database"],
    model=OpenAIChat(id='gpt-4'),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)


formula_agent = Agent(
    description='A math expert in mathematical formula',
    name="Formula Agent",
    role='Retrive formula from the documents ',
    instructions=["Provide formula based on query from the vector database",
                  "Re-arrange the formula if needed"],
    model=OpenAIChat(id='gpt-4'),
    knowledge=knowledge_base,
    add_context=True,
    markdown=True,

)

latex_agent = Agent(
    description='An expert in re-writing formula in Latex format',
    name="Latex Agent", 
    role="Write the formula in Latex format", 
    instructions=["Provide formula in Latex format"],
    markdown=True,
)



python_agent = Agent(
    description='Python developer expert in writing code',
    name="Python Agent",
    role="write and run the python code from the formula",
    model=OpenAIChat(id='gpt-4'),
    instructions=["Write the python code from the formula",
                  "Use the numbers from the query as input to the code",
                  "Run the python code that you have written",
                  "Tell the answer"],
    tools=[PythonTools(
        run_code=True)],
    show_tool_calls=True
)

answering_agent = Agent(
    name="Answering Agent",
    role="You are the head of the reservoir engineering team, who gets answer from the reservoir engineer and provide the calculation from the formula agent, if needed based on the contextual question",
    instructions=[
        "1. Retrieve information based on query from vector database",
        "2. When answers have formula give to formula agent to write the formula. If the questions need to re-arrange the formula please do so.",
        "3. Route to latex agent to convert the formula to latex format",
        "4. When the question/query have numerical inputs, pass the formula to python_agent to write the code to answers the query, by plugging those input to the python code",
        "5. Always write and run python code when asked to calculate or estimate"],
    team=[knowledge_agent, formula_agent, latex_agent, python_agent],
    show_tool_calls=True,
    markdown=True,
)

# # response = answering_agent.run("What is the STOIP of oil reservoir with A=200 acres, porosity=0.1, h=50 ft, sw=0.3, Bo=1.27 with volumetric method?")
# response=answering_agent.run("What is A in acre if STOIP is 4276062.99 barrels, porosity is 0.1, h is 50 ft, sw is 0.3, Bo is 1.27 with volumetric method?")
# pprint_run_response(response, markdown=True)


# Streamlit UI
st.set_page_config(page_title="Reservoir Engineering AI", page_icon=":earth_americas:", layout="wide")
st.title("Reservoir Engineering AI")
#set up the sidebar
st.sidebar.title("Ask a question")
#set an area for the user to input the question
question = st.sidebar.text_area("Type your question here")

if st.sidebar.button("Ask"):
    response: RunResponse = knowledge_agent.run(question)
    st.write(response.content)