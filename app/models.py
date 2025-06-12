# models.py

from pydantic import BaseModel
from typing import Optional, List


class QueryInput(BaseModel):
    question: str
    inverter_id: Optional[str] = None
    start_date: Optional[str] = None  # e.g., "2024-01-01"
    end_date: Optional[str] = None    # e.g., "2024-02-01"
    aggregation: Optional[str] = None  # e.g., "H" for hourly, "D" for daily


class InverterMetric(BaseModel):
    inverter_id: str
    value: float
    metric: str


class AggregatedResult(BaseModel):
    results: List[InverterMetric]
    description: Optional[str] = None
