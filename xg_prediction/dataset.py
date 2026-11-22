"""Download StatsBomb open data and prepare the shot-level train/test tables."""

from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from statsbombpy import sb
from tqdm import tqdm
import typer

from xg_prediction.config import RAW_DATA_DIR

app = typer.Typer()

# Bundesliga 2015/16 in the StatsBomb open data catalogue.
COMPETITION_ID = 9
SEASON_ID = 27

# Shot-level columns kept from the flattened StatsBomb event table. The outcome and
# StatsBomb's own xG value are deliberately excluded: they are the target and a leak.
RAW_COLUMNS = [
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "type",
    "possession",
    "possession_team",
    "play_pattern",
    "team",
    "player",
    "position",
    "location",
    "duration",
    "related_events",
    "match_id",
    "shot_key_pass_id",
    "shot_type",
    "shot_first_time",
    "shot_technique",
    "shot_body_part",
    "shot_freeze_frame",
    "possession_team_id",
    "team_id",
    "player_id",
    "under_pressure",
    "shot_aerial_won",
    "shot_one_on_one",
    "shot_deflected",
    "shot_open_goal",
    "shot_redirect",
    "off_camera",
    "shot_follows_dribble",
]
TARGET = "is_goal"


def download_shots() -> pd.DataFrame:
    """Download all shots of the configured season from the StatsBomb open data."""
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    logger.info(f"Downloading events for {len(matches)} matches...")

    frames = []
    for match_id in tqdm(sorted(matches["match_id"])):
        events = sb.events(match_id=match_id)
        shots = events[events["type"] == "Shot"].copy()
        # Penalty shootouts (period 5) are not part of open play or regular penalties.
        shots = shots[shots["period"] != 5]
        shots[TARGET] = (shots["shot_outcome"] == "Goal").astype(int)
        shots["match_id"] = match_id
        frames.append(shots)

    shots = pd.concat(frames, ignore_index=True)
    return shots.reindex(columns=RAW_COLUMNS + [TARGET])


def split_shots(
    shots: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split into a training and a hold-out table, each with an `id`."""
    shots = shots.copy()
    shots.insert(0, "id", range(len(shots)))
    train, test = train_test_split(
        shots, test_size=test_size, stratify=shots[TARGET], random_state=seed
    )
    return train.sort_values("id"), test.sort_values("id")


@app.command()
def main(
    output_dir: Path = RAW_DATA_DIR,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Download Bundesliga 2015/16 shots and write `train.csv` and `test.csv`."""
    shots = download_shots()
    logger.info(f"{len(shots)} shots, goal rate {shots[TARGET].mean():.3f}")

    train, test = split_shots(shots, test_size=test_size, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    logger.success(f"Wrote {len(train)} train and {len(test)} test shots to {output_dir}")


if __name__ == "__main__":
    app()
