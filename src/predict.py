"""Predict churn risk for a customer stored in a JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_churn_model.joblib"


def load_customer(path: Path) -> dict[str, object]:
    """Load one customer object from JSON."""
    with path.open(encoding="utf-8") as file:
        customer = json.load(file)
    if not isinstance(customer, dict):
        raise ValueError("The input JSON must contain one customer object.")
    return customer


def predict_customer(model_path: Path, customer: dict[str, object]) -> dict[str, object]:
    """Validate input fields and return a churn prediction."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python src/train.py"
        )

    pipeline = joblib.load(model_path)
    required_features = list(pipeline.feature_names_in_)
    missing = [feature for feature in required_features if feature not in customer]
    if missing:
        raise ValueError(f"Input is missing required fields: {missing}")

    customer_frame = pd.DataFrame([customer])[required_features]
    probability = float(pipeline.predict_proba(customer_frame)[0, 1])
    prediction = "Churn" if probability >= 0.50 else "Stay"

    if probability >= 0.67:
        risk_band = "High"
    elif probability >= 0.33:
        risk_band = "Medium"
    else:
        risk_band = "Low"

    return {
        "prediction": prediction,
        "churn_probability": probability,
        "risk_band": risk_band,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Customer JSON file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = predict_customer(arguments.model, load_customer(arguments.input))
    print(f"Prediction: {result['prediction']}")
    print(f"Churn probability: {result['churn_probability']:.1%}")
    print(f"Risk band: {result['risk_band']}")
