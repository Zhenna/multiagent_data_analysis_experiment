from app.agents import structured_agent_executor

def run_cli():
    print("🧠 Inverter Chatbot CLI. Type 'exit' to quit.")
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        result = structured_agent_executor.invoke({
        "input": user_input,
        "chat_history": [],
    })


if __name__ == "__main__":
    run_cli()
