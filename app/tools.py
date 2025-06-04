import pandas as pd
from pydantic import BaseModel
from typing import Optional
from langchain.tools import tool
from app.shared import shared_context
from app.models import QueryInput


# ----------------------
# Tool: Extract Dataset and Metric
# ----------------------

@tool(args_schema=QueryInput)
def extract_dataset_and_metric(input: QueryInput) -> str:
    """Extracts the relevant dataset and metric from the query and stores in shared context."""
    ctx = shared_context["data_context"]
    query = input.question.lower()
    selected_ds = None
    selected_metric = None

    for name, meta in ctx.items():
        if any(k in query for k in meta.get("keywords", [])):
            selected_ds = name
            break

    if selected_ds:
        for metric in ctx[selected_ds].get("metrics", []):
            if metric.lower() in query:
                selected_metric = metric
                break

    if not selected_ds:
        return "Unable to determine relevant dataset."

    shared_context["active_dataset"] = selected_ds
    shared_context["active_metric"] = selected_metric or ctx[selected_ds].get("metrics", [])[0]
    return f"Using dataset '{selected_ds}' and metric '{shared_context['active_metric']}'"


# ----------------------
# Tool: Aggregate Metric
# ----------------------
@tool(args_schema=QueryInput)
def aggregate_metric(input: QueryInput) -> str:
    """Aggregate inverter performance metrics with optional filters and custom aggregation level."""
    ds_name = shared_context.get("active_dataset", "performance")
    df = shared_context["data_sources"].get(ds_name)
    metric = shared_context["data_context"][ds_name]["metrics"][0]

    if df is None or df.empty:
        return "No data to process."

    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if input.start_date:
        df = df[df["timestamp"] >= pd.to_datetime(input.start_date)]
    if input.end_date:
        df = df[df["timestamp"] <= pd.to_datetime(input.end_date)]
    if input.inverter_id:
        df = df[df["inverter_id"] == input.inverter_id.upper()]

    df.set_index("timestamp", inplace=True)
    aggregation_level = input.aggregation.upper() if input.aggregation else "H"
    df_agg = df.groupby("inverter_id")[metric.lower()].resample(aggregation_level).mean().reset_index()

    shared_context["aggregated_df"] = df_agg
    shared_context["active_metric"] = metric.lower()
    return f"Aggregated '{metric}' by '{aggregation_level}' for inverter(s)."


class EmptyInput(BaseModel):
    pass


# ----------------------
# Tool: Final Response
# ----------------------
@tool(args_schema=EmptyInput)
def final_response_tool(_: EmptyInput) -> str:
    """Generate a concise answer from shared_context outputs."""
    result = shared_context.get("calculation_result")
    if not result:
        return "No result computed."

    inv = result["inverter_id"]
    val = result["value"]
    metric = result["metric"]

    return f"Inverter {inv} has the worst performance based on '{metric}' with a value of {val:.4f}."


# Register all tools
tools = [
    extract_dataset_and_metric,
    aggregate_metric,
    final_response_tool
]
