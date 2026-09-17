"""Train and evaluate telecom customer churn classification models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
DEFAULT_MODELS = PROJECT_ROOT / "models"

EXPECTED_COLUMNS = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the raw CSV, clean values, and separate features from the target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python src/download_data.py"
        )

    data = pd.read_csv(path)
    missing = EXPECTED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges", "Churn"]).reset_index(drop=True)

    unexpected_targets = set(data["Churn"].unique()).difference({"Yes", "No"})
    if unexpected_targets:
        raise ValueError(f"Unexpected Churn values: {sorted(unexpected_targets)}")

    customer_ids = data.pop("customerID")
    target = data.pop("Churn").map({"No": 0, "Yes": 1}).astype(int)
    return data, target, customer_ids


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create separate numeric and categorical preprocessing pipelines."""
    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def candidate_models() -> dict[str, object]:
    """Return the three models compared in this project."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def score_predictions(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Calculate classification metrics from held-out test predictions."""
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def save_churn_distribution(target: pd.Series, path: Path) -> None:
    counts = target.map({0: "Stayed", 1: "Churned"}).value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    plt.title("Customer Churn Distribution")
    plt.xlabel("")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_model_comparison(metrics: pd.DataFrame, path: Path) -> None:
    chart_data = metrics.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=chart_data, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model Performance on the Test Set")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_feature_importance(pipeline: Pipeline, path: Path, top_n: int = 15) -> None:
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        ranking = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "magnitude": np.abs(coefficients),
            }
        ).nlargest(top_n, "magnitude")
        ranking = ranking.sort_values("coefficient")
        values = ranking["coefficient"]
        colors = np.where(values > 0, "#c44e52", "#4c72b0")
        title = f"Top {top_n} Logistic Regression Churn Signals"
        x_label = "Coefficient (positive = higher predicted churn risk)"
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        ranking = (
            pd.DataFrame({"feature": feature_names, "importance": importance})
            .nlargest(top_n, "importance")
            .sort_values("importance")
        )
        values = ranking["importance"]
        colors = "#4c72b0"
        title = f"Top {top_n} Churn Predictors"
        x_label = "Model importance"
    else:
        return

    ranking["feature"] = (
        ranking["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )

    plt.figure(figsize=(9, 6))
    plt.barh(ranking["feature"], values, color=colors)
    if hasattr(model, "coef_"):
        plt.axvline(0, color="black", linewidth=0.8)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA,
    reports_dir: Path = DEFAULT_REPORTS,
    models_dir: Path = DEFAULT_MODELS,
) -> pd.DataFrame:
    """Run the full training pipeline and persist reproducible outputs."""
    sns.set_theme(style="whitegrid")
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    features, target, customer_ids = load_and_clean_data(data_path)
    save_churn_distribution(target, figures_dir / "churn_distribution.png")

    row_indices = np.arange(len(features))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    test_outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    plt.figure(figsize=(7, 6))
    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        results.append({"model": name, **score_predictions(y_test, predictions, probabilities)})
        fitted_models[name] = pipeline
        test_outputs[name] = (predictions, probabilities)
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=plt.gca())

    plt.plot([0, 1], [0, 1], "k--", label="Random chance")
    plt.title("ROC Curves on the Test Set")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=160)
    plt.close()

    metrics = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = str(metrics.loc[0, "model"])
    best_pipeline = fitted_models[best_name]
    best_predictions, best_probabilities = test_outputs[best_name]

    save_model_comparison(metrics, figures_dir / "model_comparison.png")
    save_feature_importance(best_pipeline, figures_dir / "top_features.png")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    metrics.to_csv(reports_dir / "model_metrics.csv", index=False)
    summary = {
        "dataset_rows_after_cleaning": int(len(features)),
        "training_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "positive_class": "Churn = Yes",
        "selection_metric": "roc_auc",
        "best_model": best_name,
        "models": json.loads(metrics.to_json(orient="records")),
    }
    (reports_dir / "model_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    predictions_table = pd.DataFrame(
        {
            "customerID": customer_ids.iloc[test_indices].to_numpy(),
            "actual_churn": y_test.map({0: "No", 1: "Yes"}).to_numpy(),
            "predicted_churn": np.where(best_predictions == 1, "Yes", "No"),
            "churn_probability": np.round(best_probabilities, 4),
        }
    ).sort_values("churn_probability", ascending=False)
    predictions_table.to_csv(reports_dir / "test_predictions.csv", index=False)

    joblib.dump(best_pipeline, models_dir / "best_churn_model.joblib")

    print(f"Clean rows: {len(features):,}")
    print(f"Training rows: {len(train_indices):,} | Test rows: {len(test_indices):,}")
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"Outputs saved under: {reports_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_and_evaluate(arguments.data, arguments.reports_dir, arguments.models_dir)
