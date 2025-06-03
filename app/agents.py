from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools import extract_dataset_and_metric, aggregate_metric, calculate_top_performer, sort_inverters
from app.shared import shared_context

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a data analysis assistant working with pandas dataframes (not SQL databases). "
        "Use tools when needed and always respond in natural language.\n"
        + shared_context.get("planner_prompt", "")
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

tools = [extract_dataset_and_metric, aggregate_metric, calculate_top_performer, sort_inverters]

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

structured_agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
