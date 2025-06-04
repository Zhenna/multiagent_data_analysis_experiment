# Inverter Performance Chatbot

This repository implements a multi-agent chatbot that analyzes inverter performance data, including predicted downtime, failure probabilities, and feature importance. It supports both CLI and REST API access, powered by LangChain and Ollama LLM.

---

## Features

- 🔍 Understands natural language queries like:
  - "Which inverter has the highest hourly failure probability?"
  - "Rank inverters by predicted downtime in May 2024"
  - "What factors contribute most to inverter failure?"

- 📊 Dynamically selects relevant dataset (`performance`, `features`)
- 🧠 Aggregates and analyzes metrics using Python tools
- 🧾 Converts numeric results into concise natural language responses
- 💬 Supports CLI and REST API usage

---

## Datasets

Place your data files in `data/`:

- `inverter_performance_prediction.csv`
- `feature_importance.csv`

---

## Usage

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Start the CLI chatbot**

```bash
python app/cli.py
```

3. **Run FastAPI server**

```bash
uvicorn app.main:app --reload
```

## API Usage

**POST** `/query`

```json
{
  "question": "Which inverter has the highest failure rate in January?",
  "inverter_id": "INV001",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "aggregation": "D"
}
```

## Notes

- The model used is `llama3.2` via `ChatOllama` (Ollama must be installed and running).
- Tools are invoked via structured prompts and JSON schema validation.

---

## Structure

```
.
├── app
│   ├── agents.py         # Agent setup using LangChain with function calling
│   ├── cli.py            # CLI interface
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models for structured input/output
│   ├── shared.py         # Shared data loading and global state
│   ├── tools.py          # Registered tools callable by agent
├── data
│   ├── inverter_performance_prediction.csv
│   └── feature_importance.csv
├── requirements.txt
```

---
