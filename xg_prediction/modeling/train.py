from __future__ import annotations

import json
from pathlib import Path

import dagshub
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.xgboost
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import typer
import xgboost as xgb

app = typer.Typer()

dagshub.init(repo_owner="joscha0610", repo_name="xg_prediction", mlflow=True)

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
    "id"
    "index",
    "timestamp",
    "location",
    "related_events",
    "shot_key_pass_id",
    "shot_freeze_frame",
    "player", 
    "position"
]


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in known boolean flag columns."""
    df = df.copy()
    df[BOOL_COLS] = df[BOOL_COLS].astype("boolean").fillna(False)
    return df


def preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing: fill bools, drop unused cols, one-hot encode categoricals."""
    df = fill_missing_values(df)
    df = df.drop(columns=DROP_COLS, errors="ignore")

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def _hydra_run_name(cfg: DictConfig) -> str:
    """Nice run naming: trials in multirun, stable for single runs."""
    if HydraConfig.initialized():
        job_id = HydraConfig.get().job.id
        if job_id not in (None, "???"):
            return f"{cfg.mode}_{job_id}"
    return f"{cfg.mode}_single"


def _load_xy(cfg: DictConfig) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(to_absolute_path(cfg.data.features_train_path))
    y = df[cfg.data.target_column]
    X = preprocessing_pipeline(df.drop(columns=[cfg.data.target_column]))
    return X, y


def _log_feature_columns(X: pd.DataFrame) -> None:
    cols = list(X.columns)
    path = Path("feature_columns.json")
    path.write_text(json.dumps(cols))
    mlflow.log_artifact(str(path))

def get_best_params(
    experiment_name: str = "xG_Optuna_Optimization",
) -> dict:
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError("Experiment not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.mode = 'tune'",
        order_by=["metrics.auc_mean DESC"],
        max_results=1,
    )

    best_run = runs[0]
    params = dict(best_run.data.params)

    return params, best_run.info.run_id


def _hydra_run_name(cfg) -> str:
    try:
        job_id = HydraConfig.get().job.id
    except Exception:
        job_id = "single"
    return f"{cfg.mode}_{job_id}"

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    logger.info(f"Loading data from {cfg.data.features_train_path}...")
    X, y = _load_xy(cfg)

    mlflow.set_experiment(cfg.experiment_name)
    run_name = _hydra_run_name(cfg)

    params = OmegaConf.to_container(cfg.model.params, resolve=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params({"mode": cfg.mode, "seed": cfg.seed})

        # -----------------------
        # MODE 1: Optuna tuning
        # -----------------------
        if cfg.mode == "tune":
            cv = StratifiedKFold(
                n_splits=cfg.cv.n_splits,
                shuffle=True,
                random_state=cfg.seed,
            )

            aucs: list[float] = []

            for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                proba = model.predict_proba(X_va)[:, 1]
                auc = float(roc_auc_score(y_va, proba))
                aucs.append(auc)

                mlflow.log_metric(f"auc_fold_{fold}", auc)

            auc_mean = float(np.mean(aucs))
            auc_std = float(np.std(aucs))

            mlflow.log_metric("auc_mean", auc_mean)
            mlflow.log_metric("auc_std", auc_std)

            logger.success(
                f"Logged {run_name} (run_id={run.info.run_id}) "
                f"auc_mean={auc_mean:.5f} ± {auc_std:.5f}"
            )

            # Optuna maximizes this
            return auc_mean

        # -----------------------
        # MODE 2: Final model
        # -----------------------
        if cfg.mode == "final":
            logger.info(f"Training FINAL model on all {len(X)} samples...")
            params = OmegaConf.to_container(cfg.model.params, resolve=True)
            params.pop("early_stopping_rounds", None)
            model = xgb.XGBClassifier(**params)
            model.fit(X, y, verbose=False)

            # Log model artifact + schema helpers for inference
            _log_feature_columns(X)
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="model",
                input_example=X.head(3),
            )

            logger.success(
                f"Final model logged (run_id={run.info.run_id}). "
                f"Use this run_id for predict."
            )
            return 0.0

        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()


























# from pathlib import Path

# import dagshub
# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import to_absolute_path
# from loguru import logger
# import mlflow
# import mlflow.xgboost
# import numpy as np
# from omegaconf import DictConfig, OmegaConf
# import pandas as pd
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold, train_test_split
# from tqdm import tqdm
# import typer
# import xgboost as xgb
# from xgboost.callback import EarlyStopping

# from xg_prediction.config import MODELS_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()

# dagshub.init(repo_owner='joscha0610', repo_name='xg_prediction', mlflow=True)


# BOOL_COLS = [
#     "shot_first_time",
#     "under_pressure",
#     "shot_aerial_won",
#     "shot_one_on_one",
#     "shot_deflected",
#     "shot_open_goal",
#     "shot_redirect",
#     "off_camera",
#     "shot_follows_dribble",
# ]

# DROP_COLS = [
#     "index",
#     "timestamp",
#     "location",
#     "related_events",
#     "shot_key_pass_id",
#     "shot_freeze_frame",
# ]


# def fill_missing_values(df):
#     """Fills boolean columns with False and handles other NAs."""
#     # Fill specific boolean flags with False
#     df[BOOL_COLS] = (
#     df[BOOL_COLS]
#     .astype("boolean")   # <-- critical line
#     .fillna(False)
#     )

#     return df


# def preprocessing_pipeline(df):
#     """Executes the full preprocessing flow."""
#     # 1. Fill missing values
#     df = fill_missing_values(df)

#     # 2. Drop unnecessary columns
#     # errors='ignore' ensures it won't crash if a column was already dropped
#     df = df.drop(columns=DROP_COLS, errors="ignore")

#     # 3. One-hot encoding for categorical columns
#     # This automatically detects 'object' or 'category' types
#     categorical_cols = df.select_dtypes(include=["object", "category"]).columns
#     df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

#     return df

# @hydra.main(version_base=None, config_path="../../conf", config_name="config")
# def main(cfg: DictConfig) -> float:
#     logger.info(f"Loading data from {cfg.data.features_train_path}...")
#     df = pd.read_csv(to_absolute_path(cfg.data.features_train_path))

#     target_col = cfg.data.target_column
#     y = df[target_col]
#     X = preprocessing_pipeline(df.drop(columns=[target_col]))

#     mlflow.set_experiment(cfg.experiment_name)

#     # Helpful naming: Hydra job number corresponds to Optuna trial index
#     hc = HydraConfig.get()
#     run_name = f"trial_{hc.job.id}"
#     run_suffix = "single"
#     if HydraConfig.initialized():
#         rid = HydraConfig.get().job.id
#         if rid not in (None, "???"):
#             run_suffix = rid

#     params = OmegaConf.to_container(cfg.model.params, resolve=True)

#     cv = StratifiedKFold(
#         n_splits=cfg.cv.n_splits,
#         shuffle=True,
#         random_state=cfg.seed,
#     )

#     aucs = []

#     with mlflow.start_run(run_name=run_name) as run:
#         mlflow.log_params(params)
#         mlflow.log_params({"cv_n_splits": cfg.cv.n_splits, "seed": cfg.seed})

#         for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
#             X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
#             y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

#             model = xgb.XGBClassifier(**params)

#             # Early stopping improves tuning quality and reduces runtime
#             model.fit(
#                 X_tr, y_tr,
#                 eval_set=[(X_va, y_va)],
#                 verbose=False
#             )

#             proba = model.predict_proba(X_va)[:, 1]
#             auc = roc_auc_score(y_va, proba)
#             aucs.append(auc)

#             mlflow.log_metric(f"auc_fold_{fold}", float(auc))

#         auc_mean = float(np.mean(aucs))
#         auc_std = float(np.std(aucs))
#         mlflow.log_metric("auc_mean", auc_mean)
#         mlflow.log_metric("auc_std", auc_std)

#         logger.success(
#     f"Logged trial {run_suffix} to MLflow "
#     f"(run_id={run.info.run_id}) "
#     f"auc_mean={auc_mean:.5f} ± {auc_std:.5f}"
# )

#     # Optuna will maximize this value
#     return auc_mean


# if __name__ == "__main__":
#     main()