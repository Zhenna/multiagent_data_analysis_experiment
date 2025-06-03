import pandas as pd

data_sources = {
    "performance": pd.read_csv("data/inverter_performance_prediction.csv"),
    "features": pd.read_csv("data/feature_importance.csv")
}

shared_context = {}

# Add dataset-level context for intelligent routing
shared_context["data_context"] = {
    "performance": {
        "description": "Inverter-level performance metrics, including timestamps and downtime indicators.",
        "key_columns": ["timestamp", "inverter_id"],
        "metrics": ["target downtime", "predicted downtime", "predicted failure probabilities"]
    },
    "features": {
        "description": "Feature importance scores for predictions of downtime and failure probability, indicating model relevance per feature.",
        "key_columns": ["feature_name" ,"importance"],
        "metrics": ["importance"]
    }
}
