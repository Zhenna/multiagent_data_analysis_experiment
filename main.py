from fastapi import FastAPI
from pydantic import BaseModel
from app.agents import structured_agent_executor
import uvicorn
import sys

# ----- REST API -----
app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.post("/query")
def query_agent(input: QueryInput):
    result = structured_agent_executor.invoke({"input": input.query})
    return {"result": result}

# ----- CLI entry -----
def run_cli():
    print("🧠 Inverter Chatbot CLI. Type 'exit' to quit.")
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        result = structured_agent_executor.invoke({
            "input": input.query,
            "chat_history": [],  # Add this line
        })
        print(result)

# ----- Entrypoint -----
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
