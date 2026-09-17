"""Small tests for the churn data-cleaning and evaluation helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import load_and_clean_data, score_predictions


class ChurnPipelineTests(unittest.TestCase):
    def test_load_and_clean_data(self) -> None:
        sample = pd.DataFrame(
            {
                "customerID": ["A", "B", "C"],
                "tenure": [1, 12, 2],
                "MonthlyCharges": [20.0, 80.0, 50.0],
                "TotalCharges": ["20.0", "960.0", " "],
                "Contract": ["Month-to-month", "One year", "Month-to-month"],
                "Churn": ["Yes", "No", "Yes"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            sample.to_csv(path, index=False)
            features, target, customer_ids = load_and_clean_data(path)

        self.assertEqual(len(features), 2)
        self.assertIn(features["TotalCharges"].dtype.kind, "fi")
        self.assertEqual(target.tolist(), [1, 0])
        self.assertEqual(customer_ids.tolist(), ["A", "B"])

    def test_score_predictions(self) -> None:
        actual = pd.Series([0, 0, 1, 1])
        predicted = np.array([0, 1, 1, 1])
        probabilities = np.array([0.1, 0.7, 0.8, 0.9])

        metrics = score_predictions(actual, predicted, probabilities)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
