from pathlib import Path

import dagshub
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from loguru import logger
import mlflow
import mlflow.xgboost
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
import typer
import xgboost as xgb
from xgboost.callback import EarlyStopping

from xg_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

dagshub.init(repo_owner='joscha0610', repo_name='xg_prediction', mlflow=True)


BOOL_COLS = [
    "shot_first_time",
    "under_pressure",
    "shot_aerial_won",
    "shot_one_on_one",
    "shot_deflected",
    "shot_open_goal",
    "shot_redirect",
    "off_camera",
    "shot_follows_dribble",
]

DROP_COLS = [
    "index",
    "timestamp",
    "location",
    "related_events",
    "shot_key_pass_id",
    "shot_freeze_frame",
]


def fill_missing_values(df):
    """Fills boolean columns with False and handles other NAs."""
    # Fill specific boolean flags with False
    df[BOOL_COLS] = (
    df[BOOL_COLS]
    .astype("boolean")   # <-- critical line
    .fillna(False)
    )

    return df


def preprocessing_pipeline(df):
    """Executes the full preprocessing flow."""
    # 1. Fill missing values
    df = fill_missing_values(df)

    # 2. Drop unnecessary columns
    # errors='ignore' ensures it won't crash if a column was already dropped
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # 3. One-hot encoding for categorical columns
    # This automatically detects 'object' or 'category' types
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    logger.info(f"Loading data from {cfg.data.features_train_path}...")
    df = pd.read_csv(to_absolute_path(cfg.data.features_train_path))

    target_col = cfg.data.target_column
    y = df[target_col]
    X = preprocessing_pipeline(df.drop(columns=[target_col]))

    mlflow.set_experiment(cfg.experiment_name)

    # Helpful naming: Hydra job number corresponds to Optuna trial index
    hc = HydraConfig.get()
    run_name = f"trial_{hc.job.id}"
    run_suffix = "single"
    if HydraConfig.initialized():
        rid = HydraConfig.get().job.id
        if rid not in (None, "???"):
            run_suffix = rid

    params = OmegaConf.to_container(cfg.model.params, resolve=True)

    cv = StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )

    aucs = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params({"cv_n_splits": cfg.cv.n_splits, "seed": cfg.seed})

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = xgb.XGBClassifier(**params)

            # Early stopping improves tuning quality and reduces runtime
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )

            proba = model.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, proba)
            aucs.append(auc)

            mlflow.log_metric(f"auc_fold_{fold}", float(auc))

        auc_mean = float(np.mean(aucs))
        auc_std = float(np.std(aucs))
        mlflow.log_metric("auc_mean", auc_mean)
        mlflow.log_metric("auc_std", auc_std)

        logger.success(
    f"Logged trial {run_suffix} to MLflow "
    f"(run_id={run.info.run_id}) "
    f"auc_mean={auc_mean:.5f} ± {auc_std:.5f}"
)

    # Optuna will maximize this value
    return auc_mean


if __name__ == "__main__":
    main()

# @hydra.main(version_base=None, config_path="../../conf", config_name="config")
# def main(cfg: DictConfig) -> float:
#     # 1. Load Data
#     logger.info(f"Loading data from {cfg.data.features_path}...")
#     df = pd.read_csv(to_absolute_path(cfg.data.features_path))
    
#     # 2. Split Features and Labels
#     target_col = cfg.data.target_column
#     X_train = preprocessing_pipeline(df.drop(columns=[target_col]))
#     y_train = df[target_col]

#     # 3. MLflow Tracking Setup
#     mlflow.set_experiment(cfg.experiment_name)
    
#     with mlflow.start_run() as run:
#         # Log hyperparameters from Hydra
#         params = OmegaConf.to_container(cfg.model.params, resolve=True)
#         mlflow.log_params(params)
        
#         # 4. Train Model on 100% of data
#         logger.info(f"Training XGBoost on all {len(X_train)} samples...")
#         model = xgb.XGBClassifier(**params)
#         model.fit(X_train, y_train, verbose=False)

#         # 5. Log Model to MLflow (No local saving needed!)
#         # This saves the model in DagsHub's artifact store
#         mlflow.xgboost.log_model(
#             xgb_model=model, 
#             artifact_path="model",
#             input_example=X_train.head(3) # Useful for tracking schema
#         )
        
#         logger.success(f"Model logged to MLflow. Run ID: {run.info.run_id}")

#     # Return 0.0 because there is no validation set to evaluate
#     return 0.0

# if __name__ == "__main__":
#     main()