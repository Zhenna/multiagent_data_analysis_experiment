from app.agents import planner_executor
from app.models import QueryInput

def run_cli():
    print("Welcome to the Inverter Analysis Chatbot!")
    print("Type 'exit' to quit.")
    while True:
        question = input("\n> ")
        if question.lower() in ("exit", "quit"): break
        user_input = QueryInput(question=question)
        result = planner_executor.invoke(user_input.model_dump())

        answer = result.get("output")
        steps = result.get("intermediate_steps", [])

        print("\nAnswer:", answer)
        if steps:
            print("\n[Toolchain Execution Log]")
            for step in steps:
                print("-", step[0].tool, step[1])

if __name__ == "__main__":
    run_cli()
