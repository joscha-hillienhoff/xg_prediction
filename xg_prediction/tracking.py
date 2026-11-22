"""Experiment tracking setup: local MLflow by default, DagsHub when explicitly enabled."""

from loguru import logger
import mlflow

from xg_prediction.config import PROJ_ROOT

DAGSHUB_REPO_OWNER = "joscha0610"
DAGSHUB_REPO_NAME = "xg_prediction"
LOCAL_TRACKING_URI = (PROJ_ROOT / "mlruns").as_uri()


def init_tracking(remote: bool = False) -> None:
    """Point MLflow at `./mlruns`, or at the DagsHub remote when `remote` is true.

    The remote requires DagsHub credentials (an interactive login or the
    `DAGSHUB_USER_TOKEN` environment variable); none are stored in the repository.
    """
    if remote:
        import dagshub

        logger.info(f"Tracking to DagsHub repo {DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}")
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    else:
        logger.info(f"Tracking locally to {LOCAL_TRACKING_URI}")
        mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
