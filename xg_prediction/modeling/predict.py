from pathlib import Path

import dagshub
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from tqdm import tqdm
import typer

from xg_prediction.config import PROCESSED_DATA_DIR
from xg_prediction.modeling import train

app = typer.Typer()

# Initialize dagshub to ensure the script can reach the remote server
dagshub.init(repo_owner='joscha0610', repo_name='xg_prediction', mlflow=True)

def load_model(run_id: str):
    """Load the logged XGBoost model from an MLflow run."""
    mlflow.set_tracking_uri("https://dagshub.com/joscha0610/xg_prediction.mlflow")
    mlflow.set_registry_uri("https://dagshub.com/joscha0610/xg_prediction.mlflow")
    model_uri = f"runs:/{run_id}/model"
    return mlflow.xgboost.load_model(model_uri)


def predict_proba(model, X: pd.DataFrame) -> pd.Series:
    """Return predicted goal probabilities."""
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index, name="is_goal")


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
    model = load_model("7257190dfa8a445e90176dafcd01b166")

    # Predict
    preds = predict_proba(model, X_test)

    # Save submission
    create_submission(submission_path, ids=ids, predictions=preds)

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()
