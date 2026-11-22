from __future__ import annotations

import json
from pathlib import Path

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
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import typer
import xgboost as xgb

from xg_prediction.tracking import init_tracking

app = typer.Typer()

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
    "id",
    "index",
    "timestamp",
    "location",
    "related_events",
    "shot_key_pass_id",
    "shot_freeze_frame",
    "player",
    "position",
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


def _save_oof(oof: pd.Series, path: str) -> None:
    """Write out-of-fold probabilities (one per training row) and log them as an artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out)
    mlflow.log_artifact(str(out))


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

    init_tracking(remote=cfg.tracking.remote)
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
            log_losses: list[float] = []
            oof = pd.Series(np.nan, index=X.index, name="xg")

            for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                proba = model.predict_proba(X_va)[:, 1]
                auc = float(roc_auc_score(y_va, proba))
                aucs.append(auc)
                oof.iloc[va_idx] = proba

                ll = float(log_loss(y_va, proba))
                log_losses.append(ll)

                mlflow.log_metric(f"auc_fold_{fold}", auc)
                mlflow.log_metric(f"logloss_fold_{fold}", ll)

            auc_mean = float(np.mean(aucs))
            auc_std = float(np.std(aucs))

            mlflow.log_metric("auc_mean", auc_mean)
            mlflow.log_metric("auc_std", auc_std)
            mlflow.log_metric("logloss_mean", float(np.mean(log_losses)))
            mlflow.log_metric("logloss_std", float(np.std(log_losses)))

            if cfg.data.oof_path:
                _save_oof(oof, to_absolute_path(cfg.data.oof_path))

            logger.success(
                f"Logged {run_name} (run_id={run.info.run_id}) "
                f"auc_mean={auc_mean:.5f} ± {auc_std:.5f} "
                f"logloss_mean={np.mean(log_losses):.5f}"
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
                registered_model_name="xg_boost",
                input_example=X.head(3),
            )

            logger.success(
                f"Final model logged (run_id={run.info.run_id}) and registered as 'xg_boost'."
            )
            return 0.0

        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
