from pathlib import Path

from loguru import logger
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import typer

from xg_prediction.config import PROCESSED_DATA_DIR
from xg_prediction.modeling import train
from xg_prediction.tracking import init_tracking

app = typer.Typer()


def load_model(model_name: str, model_version: str):
    """Load a registered XGBoost model, e.g. `load_model("xg_boost", "latest")`."""
    return mlflow.xgboost.load_model(f"models:/{model_name}/{model_version}")


def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    # binary:logistic -> returns probability as 1D array
    proba = model.predict_proba(X)[:, 1]  # probability of class 1
    proba = np.asarray(proba, dtype=float)
    return pd.Series(proba, index=X.index, name="is_goal")


def create_submission(path: Path, ids: pd.Series, predictions: pd.Series) -> None:
    """Create Kaggle submission CSV."""
    submission = pd.DataFrame({"id": ids, "is_goal": predictions})
    # Kaggle usually accepts full precision; rounding can hurt slightly
    submission.to_csv(path, index=False)
    logger.success(f"Saved submission to: {path}")


@app.command()
def main(
    test_path: Path = PROCESSED_DATA_DIR / "test_feat.csv",
    submission_path: Path = PROCESSED_DATA_DIR / "submission.csv",
    model_name: str = "xg_boost",
    model_version: str = "latest",
    remote_tracking: bool = typer.Option(
        False, help="Load the model from the DagsHub registry instead of ./mlruns."
    ),
):
    init_tracking(remote=remote_tracking)

    logger.info(f"Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)

    if "id" not in df_test.columns:
        raise ValueError("Test file must contain an 'id' column for submission.")

    ids = df_test["id"].copy()

    # Test data usually does NOT have is_goal; drop only if present.
    drop_cols = ["is_goal"]
    X_test_raw = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    # Apply same preprocessing as training
    X_test = train.preprocessing_pipeline(X_test_raw)

    model = load_model(model_name, model_version)

    # One-hot encoding can yield different columns than in training; align to the model.
    X_test = X_test.reindex(columns=model.get_booster().feature_names, fill_value=0)

    # Predict
    preds = predict_proba(model, X_test)

    # Save submission
    create_submission(submission_path, ids=ids, predictions=preds)

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
