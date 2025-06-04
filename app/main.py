from fastapi import FastAPI
from app.agents import planner_executor
from app.models import QueryInput

app = FastAPI(title="Inverter Chatbot API")

@app.post("/query")
def query_handler(payload: QueryInput):
    result = planner_executor.invoke(payload.model_dump())
    return {
        "answer": result.get("output"),
        "intermediate_steps": result.get("intermediate_steps", [])
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Inverter chatbot is running."}
