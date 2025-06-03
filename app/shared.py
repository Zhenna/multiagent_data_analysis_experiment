import os
import pandas as pd

# Define explicit paths to CSV files
DATA_PATHS = {
    "performance": "data/inverter_performance_prediction.csv",
    "features": "data/feature_importance.csv",
    # "sensors": "data/inverter_sensor_history.csv"  # Optional
}

# Load datasets
data_sources = {
    name: pd.read_csv(path)
    for name, path in DATA_PATHS.items()
    if os.path.exists(path)
}

# Initialize global shared context
shared_context = {
    "data_sources": data_sources,
    "data_context": {}
}

# Manual metadata for known datasets
predefined_context = {
    "performance": {
        "description": "Predicted and actual inverter downtimes and failure probabilities.",
        "keywords": ["downtime", "failure", "probability", "performance"],
        "key_columns": ["timestamp", "inverter_id"],
        "metrics": ["target downtime", "predicted downtime", "predicted failure probabilities"]
    },
    "features": {
        "description": "Feature importance scores for predicting inverter performance.",
        "keywords": ["feature", "importance", "factor", "variable", "contributor"],
        "key_columns": ["feature_name"],
        "metrics": ["importance"]
    }
}

# Enrich context with schema information
for name, df in data_sources.items():
    df.columns = [col.lower() for col in df.columns]
    context = predefined_context.get(name, {})
    context["metrics"] = list(set(context.get("metrics", []) + list(df.columns)))
    context["datatypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    context["datetime_cols"] = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    context["preview"] = df.head(1).to_dict(orient="records")[0] if not df.empty else {}
    shared_context["data_context"][name] = context

# Generate planner prompt
planner_description = "\n".join(
    f"- {name}: {cfg['description']}" for name, cfg in shared_context["data_context"].items()
)
shared_context["planner_prompt"] = "You can choose from the following datasets:\n" + planner_description
