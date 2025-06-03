# LangChain Master-Agent Inverter Failure Analyzer (Local Ollama)
Uses LangChain + Ollama with a master agent that decides whether to aggregate or calculate based on the natural language prompt.

## ✅ Setup
```bash
pip install -r requirements.txt
ollama pull llama3.2
```

## ✅ Run API
```bash
uvicorn app.main:app --reload
```

POST to `/run` with:
```json
{ "query": "Which inverter has the highest predicted downtime by hour?" }
```

## ✅ Run CLI
```bash
python app/cli.py
```
You will be prompted to enter your query interactively.

## ✅ CSV Requirements
Your CSV must include:
- `timestamp` (format: `%Y-%m-%d %H:%M:%S`)
- `inverter_id` (str)
- one or more of:
  - `target downtime`
  - `predicted downtime`
  - `predicted failure probabilities`