from app.shared import data_sources, shared_context
import pandas as pd
import re
from langchain.tools import tool
from pydantic import BaseModel

# Dynamically generate context-aware planner prompt
if "data_context" in shared_context:
    ctx = shared_context["data_context"]
    descriptions = [f"- {name}: {cfg['description']}" for name, cfg in ctx.items() if "description" in cfg]
    planner_description = "You can choose from the following datasets:\n" + "\n".join(descriptions)
    shared_context["planner_prompt"] = planner_description

class ExtractInput(BaseModel):
    query: str

@tool
def extract_dataset_and_metric(input: ExtractInput) -> dict:
    """Infer the dataset and metric to analyze based on user query, using contextual definitions."""
    query = input.query.lower()
    context = shared_context.get("data_context", {})

    for dataset, info in context.items():
        for metric in info.get("metrics", []):
            if metric.lower() in query:
                return {"dataset": dataset, "metric": metric}

    fallback_keywords = {
        "feature": ("features", "importance"),
        "probability": ("performance", "predicted failure probabilities"),
        "downtime": ("performance", "predicted downtime"),
    }
    for k, (ds, m) in fallback_keywords.items():
        if k in query:
            return {"dataset": ds, "metric": m}

    return {"dataset": "performance", "metric": "predicted failure probabilities"}

class AggregateInput(BaseModel):
    dataset: str
    metric: str
    time_granularity: str = 'H'

@tool
def aggregate_metric(input: AggregateInput) -> str:
    """Aggregate the specified metric in the dataset by time and inverter ID."""
    dataset = input.dataset
    metric = input.metric
    time_granularity = input.time_granularity
    freq_map = {'minute': 'T', 'hour': 'H', 'day': 'D', 'week': 'W', 'month': 'M'}
    freq = freq_map.get(time_granularity.lower(), 'H')

    df_local = data_sources[dataset].copy()
    if 'timestamp' not in df_local.columns:
        return f"Dataset '{dataset}' does not contain timestamp information for aggregation."
    df_local['timestamp'] = pd.to_datetime(df_local['timestamp'], errors='coerce')
    if df_local['timestamp'].isnull().any():
        return "Some timestamp values could not be parsed."

    try:
        df_local.set_index('timestamp', inplace=True)
        aggregated = df_local.groupby('inverter_id').resample(freq)[metric].mean().reset_index()
        shared_context['aggregated_df'] = aggregated
        shared_context['metric_col'] = metric
        shared_context['dataset'] = dataset
        shared_context['time_granularity'] = freq
        return f"Averaged '{metric}' over {freq}-level time bins per inverter."
    except Exception as e:
        return f"Aggregation failed: {e}"

class CalculateInput(BaseModel):
    metric: str

@tool
def calculate_top_performer(input: CalculateInput) -> str:
    """Calculate which inverter has the worst (highest) value of the given metric."""
    metric = input.metric
    df_agg = shared_context.get('aggregated_df')
    dataset_key = shared_context.get('dataset', 'performance')
    df = data_sources[dataset_key]

    if df_agg is None:
        if metric not in df.columns:
            return f"Column '{metric}' not found."
        df_agg = df.groupby('inverter_id')[metric].mean().reset_index()

    try:
        grouped = df_agg.groupby('inverter_id')[metric].mean()
        top_inverter = grouped.idxmax()
        top_value = grouped.max()
        return f"{top_inverter} has the highest '{metric}' with an average of {top_value:.4f}."
    except Exception as e:
        return f"Calculation failed: {e}"

class SortInput(BaseModel):
    metric: str

@tool
def sort_inverters(input: SortInput) -> str:
    """Sort inverters by the given metric and return top/bottom performers."""
    metric = input.metric
    dataset_key = shared_context.get('dataset', 'performance')
    df = data_sources[dataset_key]
    df_agg = shared_context.get('aggregated_df')

    if df_agg is None:
        if metric not in df.columns:
            return f"Column '{metric}' not found."
        df_agg = df.groupby('inverter_id')[metric].mean().reset_index()

    try:
        grouped = df_agg.groupby('inverter_id')[metric].mean()
        top3 = grouped.sort_values(ascending=False).head(3).round(4)
        bottom3 = grouped.sort_values().head(3).round(4)
        top_str = ", ".join([f"{idx} ({val})" for idx, val in top3.items()])
        bottom_str = ", ".join([f"{idx} ({val})" for idx, val in bottom3.items()])
        return f"Top 3: {top_str}. Bottom 3: {bottom_str}."
    except Exception as e:
        return f"Sorting failed: {e}"
