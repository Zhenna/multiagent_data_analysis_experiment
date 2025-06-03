# app/main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.agents import planner_executor

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(req: QueryRequest):
    """POST endpoint for querying the inverter chatbot."""
    query = req.query
    response = planner_executor.invoke({"input": query})
    return {"response": response}
