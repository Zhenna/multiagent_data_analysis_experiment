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
def extract_dataset_and_metric(**kwargs) -> str:
    """Extract relevant dataset and metric from the question and update shared context."""
    input = QueryInput(**kwargs)
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
def aggregate_metric(**kwargs) -> str:
    """Aggregate inverter performance metrics with optional filters and custom aggregation level."""
    input = QueryInput(**kwargs)
    ds_name = shared_context.get("active_dataset", "performance")
    df = shared_context["data_sources"].get(ds_name)
    metric = shared_context["data_context"][ds_name]["metrics"][0]

    if df is None or df.empty:
        return "No data to process."

    df.columns = [col.strip().lower() for col in df.columns]
    metric_col = metric.lower()

    if "timestamp" not in df.columns:
        return "Dataset is missing required 'timestamp' column for aggregation."

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Clean and filter metric column
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    if input.start_date:
        df = df[df["timestamp"] >= pd.to_datetime(input.start_date)]
    if input.end_date:
        df = df[df["timestamp"] <= pd.to_datetime(input.end_date)]
    if input.inverter_id:
        df = df[df["inverter_id"] == input.inverter_id.upper()]

    if df.empty:
        return "No data available after filtering."

    df.set_index("timestamp", inplace=True)
    aggregation_level = input.aggregation.upper() if input.aggregation else "H"

    # Resample and aggregate
    df_agg = (
        df.groupby("inverter_id")[metric_col]
        .resample(aggregation_level)
        .mean()
        .reset_index()
    )

    # Compute average per inverter over time
    df_agg_by_inv = df_agg.groupby("inverter_id")[metric_col].mean().reset_index()

    shared_context["aggregated_df"] = df_agg_by_inv
    shared_context["active_metric"] = metric_col
    return f"Aggregated '{metric}' by '{aggregation_level}' for inverter(s)."


# @tool(args_schema=QueryInput)
# def aggregate_metric(**kwargs) -> str:
#     """Aggregate inverter performance metrics with optional filters and custom aggregation level."""
#     input = QueryInput(**kwargs)
#     ds_name = shared_context.get("active_dataset", "performance")
#     df = shared_context["data_sources"].get(ds_name)
#     metric = shared_context["data_context"][ds_name]["metrics"][0]

#     if df is None or df.empty:
#         return "No data to process."

#     df.columns = [col.strip().lower() for col in df.columns]

#     if "timestamp" not in df.columns:
#         return "Dataset is missing required 'timestamp' column for aggregation."

#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df = df.dropna(subset=["timestamp"])

#     if input.start_date:
#         df = df[df["timestamp"] >= pd.to_datetime(input.start_date)]
#     if input.end_date:
#         df = df[df["timestamp"] <= pd.to_datetime(input.end_date)]
#     if input.inverter_id:
#         df = df[df["inverter_id"] == input.inverter_id.upper()]

#     df.set_index("timestamp", inplace=True)
#     aggregation_level = input.aggregation.upper() if input.aggregation else "H"
#     df_agg = df.groupby("inverter_id")[metric.lower()].resample(aggregation_level).mean().reset_index()
#     df_agg_by_inv = df_agg.groupby("inverter_id")[metric.lower()].mean(numeric_only=True).reset_index()

#     shared_context["aggregated_df"] = df_agg_by_inv
#     shared_context["active_metric"] = metric.lower()
#     return f"Aggregated '{metric}' by '{aggregation_level}' for inverter(s)."


class EmptyInput(BaseModel):
    pass


# ----------------------
# Tool: Sort Inverters
# ----------------------
@tool(args_schema=EmptyInput)
def sort_inverters(**kwargs) -> str:
    """Sort inverters by average value of the selected metric from aggregated data."""
    df = shared_context.get("aggregated_df")
    metric = shared_context.get("active_metric")

    if df is None or df.empty:
        return "No aggregated data available."

    try:
        df.columns = [col.lower() for col in df.columns]
        metric = metric.lower()
        mean_vals = df.groupby("inverter_id")[metric].mean().sort_values(ascending=False)
        shared_context["sorted_inverters"] = mean_vals.to_dict()
        return f"Sorted inverters by average '{metric}':\n{mean_vals.to_string()}"
    except Exception as e:
        return f"Sorting failed: {e}"


# ----------------------
# Tool: Final Response
# ----------------------
@tool(args_schema=EmptyInput)
def final_response_tool(**kwargs) -> str:
    """Generate a concise answer from shared_context outputs."""
    sorted_result = shared_context.get("sorted_inverters")
    if not sorted_result:
        return "No sorted results to report."

    top_inverter = next(iter(sorted_result))
    return f"Top inverter by performance: {top_inverter}"


# Register all tools
tools = [
    extract_dataset_and_metric,
    aggregate_metric,
    sort_inverters,
    final_response_tool
]
