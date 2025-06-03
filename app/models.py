import pandas as pd

# Load your datasets
performance_df = pd.read_csv("data/performance_metrics.csv")  # has timestamp, inverter_id, downtime metrics
features_df = pd.read_csv("data/feature_importance.csv")      # has inverter_id, feature, importance

# Preprocess timestamps
performance_df['timestamp'] = pd.to_datetime(performance_df['timestamp'], errors='coerce')

# Global registry of datasets
data_sources = {
    "performance": performance_df,
    "features": features_df,
}
