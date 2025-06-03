# app/cli.py

import sys
from app.agents import planner_executor

def main():
    print("🔌 Inverter Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("🧠 Query: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        result = planner_executor.invoke({"input": user_input})
        print(f"🤖 Answer: {result}\n")

if __name__ == "__main__":
    main()
