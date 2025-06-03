import pandas as pd
from langchain.tools import tool
from app.shared import shared_context

# ----------------------
# Tool: Extract Dataset and Metric
# ----------------------
@tool
def extract_dataset_and_metric(query: str) -> str:
    """Extracts the relevant dataset and metric from the query and stores in shared context."""
    ctx = shared_context["data_context"]
    selected_ds = None
    selected_metric = None

    for name, meta in ctx.items():
        if any(k in query.lower() for k in meta.get("keywords", [])):
            selected_ds = name
            break

    if selected_ds:
        for metric in ctx[selected_ds].get("metrics", []):
            if metric.lower() in query.lower():
                selected_metric = metric
                break

    if not selected_ds:
        return "Unable to determine relevant dataset."

    shared_context["active_dataset"] = selected_ds
    shared_context["active_metric"] = selected_metric or ctx[selected_ds].get("metrics", [])[0]
    return f"Using dataset '{selected_ds}' and metric '{shared_context['active_metric']}'."


# ----------------------
# Tool: Aggregate Metric
# ----------------------
@tool
def aggregate_metric(query: str) -> str:
    """Aggregates the selected metric by timestamp and inverter_id, supports filtering by date or inverter_id."""
    ds_name = shared_context.get("active_dataset")
    metric = shared_context.get("active_metric")
    df = shared_context["data_sources"].get(ds_name)

    if df is None or df.empty:
        return "No data found to aggregate."

    try:
        df.columns = [col.lower() for col in df.columns]
        metric = metric.lower()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # Optional filtering
        if "start" in query.lower() or "end" in query.lower():
            if "start=" in query.lower():
                start = query.lower().split("start=")[1].split()[0]
                df = df[df["timestamp"] >= pd.to_datetime(start, errors="coerce")]
            if "end=" in query.lower():
                end = query.lower().split("end=")[1].split()[0]
                df = df[df["timestamp"] <= pd.to_datetime(end, errors="coerce")]

        if "inverter=" in query.lower():
            inv = query.lower().split("inverter=")[1].split()[0].upper()
            df = df[df["inverter_id"] == inv]

        df.set_index("timestamp", inplace=True)
        df_agg = df.groupby("inverter_id")[metric].resample("H").mean().reset_index()
        shared_context["aggregated_df"] = df_agg
        return f"Aggregated '{metric}' by hour."
    except Exception as e:
        return f"Aggregation failed: {e}"


# ----------------------
# Tool: Calculate Top Performer
# ----------------------
@tool
def calculate_top_performer(_: str) -> str:
    """Finds the worst performing inverter based on the selected metric."""
    ds_name = shared_context.get("active_dataset")
    df = shared_context.get("aggregated_df") or shared_context["data_sources"].get(ds_name)
    metric = shared_context.get("active_metric")

    if df is None or df.empty:
        return "No data available for calculation."

    try:
        df.columns = [col.lower() for col in df.columns]
        metric = metric.lower()
        worst = df.groupby("inverter_id")[metric].mean().idxmax()
        val = df.groupby("inverter_id")[metric].mean().max()
        shared_context["calculation_result"] = {
            "inverter_id": worst,
            "value": val,
            "metric": metric
        }
        return f"Worst performer: {worst} ({val:.4f})"
    except Exception as e:
        return f"Calculation failed: {e}"


# ----------------------
# Tool: Sort Inverters
# ----------------------
@tool
def sort_inverters(_: str) -> str:
    """Sort inverters by metric and store top/bottom performers."""
    ds_name = shared_context.get("active_dataset")
    df = shared_context.get("aggregated_df") or shared_context["data_sources"].get(ds_name)
    metric = shared_context.get("active_metric")

    if df is None or df.empty:
        return "No data available to sort."

    try:
        df.columns = [col.lower() for col in df.columns]
        metric = metric.lower()
        mean_vals = df.groupby("inverter_id")[metric].mean().sort_values(ascending=False)
        shared_context["sorted_inverters"] = mean_vals.to_dict()
        return f"Top 3 inverters:\n{mean_vals.head(3).to_string()}"
    except Exception as e:
        return f"Sorting failed: {e}"


# ----------------------
# Tool: Final Response
# ----------------------
@tool
def final_response_tool(_: str) -> str:
    """Generate a concise answer from shared_context outputs."""
    result = shared_context.get("calculation_result")
    if not result:
        return "No result computed."

    inv = result["inverter_id"]
    val = result["value"]
    metric = result["metric"]

    return f"Inverter {inv} has the worst performance based on '{metric}' with a value of {val:.4f}."


# ----------------------
# List of all tools
# ----------------------
tools = [
    extract_dataset_and_metric,
    aggregate_metric,
    calculate_top_performer,
    sort_inverters,
    final_response_tool
]
