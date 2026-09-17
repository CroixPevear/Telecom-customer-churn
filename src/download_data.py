"""Download and validate the public IBM Telco Customer Churn dataset."""

from __future__ import annotations

import argparse
import ssl
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

try:
    import certifi
except ImportError:  # A standard certificate store still works on most systems.
    certifi = None


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def validate_dataset(path: Path) -> pd.DataFrame:
    """Load the CSV and raise a clear error if its schema is unexpected."""
    data = pd.read_csv(path)
    missing = EXPECTED_COLUMNS.difference(data.columns)

    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Downloaded dataset is empty.")
    if not set(data["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn must contain only 'Yes' and 'No' values.")

    return data


def download_dataset(output_path: Path = DEFAULT_OUTPUT, force: bool = False) -> Path:
    """Download the CSV unless a valid local copy already exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        data = validate_dataset(output_path)
        print(f"Dataset already exists: {output_path}")
        print(f"Validated {len(data):,} rows and {len(data.columns)} columns.")
        return output_path

    print("Downloading the IBM Telco Customer Churn dataset...")
    certificate_file = certifi.where() if certifi is not None else None
    ssl_context = ssl.create_default_context(cafile=certificate_file)
    with urlopen(DATA_URL, timeout=30, context=ssl_context) as response:
        output_path.write_bytes(response.read())

    data = validate_dataset(output_path)
    print(f"Saved {len(data):,} rows and {len(data.columns)} columns to:")
    print(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when the output file already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    download_dataset(arguments.output, force=arguments.force)
