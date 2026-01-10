from pathlib import Path

import dagshub
from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from xg_prediction.config import PROCESSED_DATA_DIR
from xg_prediction.modeling import train

app = typer.Typer()

# Initialize dagshub to ensure the script can reach the remote server
dagshub.init(repo_owner="joscha0610", repo_name="xg_prediction", mlflow=True)


def load_model(model_name: str, model_version: str):
    # model_uri = f"runs:/{run_id}/model"
    model_uri = "models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)


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
    experiment_name: str = "xG_Optuna_Optimization",
):
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

    # Load model by run_id
    model = load_model("xg_boost", "2")

    # Predict
    preds = predict_proba(model, X_test)

    # Save submission
    create_submission(submission_path, ids=ids, predictions=preds)

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
