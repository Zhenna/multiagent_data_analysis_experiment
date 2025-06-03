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

### CLI

```bash
python cli.py
```

Then enter your natural language question.

### REST API

```bash
uvicorn main:app --reload
```

Send a POST request to:

```http
POST http://localhost:8000/query
{
  "query": "Which inverter has the highest failure probability in May 2024?"
}
```

---

## Structure

```
inverter_chatbot/
├── app/
│   ├── agents.py           # LangChain agent setup using tools and shared context
│   ├── cli.py              # CLI interface to interact with the chatbot
│   ├── main.py             # FastAPI app for REST API interaction
│   ├── shared.py           # Loads CSVs and stores shared_context and dataset metadata
│   └── tools.py            # Tool definitions for data extraction, aggregation, analysis
├── data/
│   ├── inverter_performance_prediction.csv
│   └── feature_importance.csv
├── README.md               # Documentation and instructions
├── requirements.txt        # Python dependencies

```

---

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`

Ensure you have [Ollama](https://ollama.com/) installed and running a model like `llama3`.

---

## Acknowledgment

Built using LangChain, FastAPI, and Ollama for local language modeling.