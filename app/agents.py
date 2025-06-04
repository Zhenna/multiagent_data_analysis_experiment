# agents.py

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from app.tools import tools
from app.shared import shared_context

from dotenv import load_dotenv
load_dotenv()

# Initialize the language model
# llm = ChatOllama(model="llama3.2", temperature=0)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Prompt to guide agent behavior and restrict to tool usage
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that analyzes inverter performance data.\n"
     "Always use the provided tools and shared context to answer user queries. Do not guess or use external knowledge.\n"
     f"{shared_context['planner_prompt']}"),
    ("human", "{question}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the structured agent
planner_agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Wrap the agent in an executor
planner_executor = AgentExecutor(
    agent=planner_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)
