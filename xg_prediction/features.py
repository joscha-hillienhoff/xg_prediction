from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import math
import pandas as pd
import ast

from xg_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
GOALPOST_Y_LOW = 36.0
GOALPOST_Y_HIGH = 44.0

def parse_location(loc: object) -> tuple[float, float] | tuple[float, float]:
    """
    Parses StatsBomb 'location' which is often stored as:
      - list [x, y]
      - string like "[101.5, 49.7]"
    Returns (x, y) as floats.
    """
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    if isinstance(loc, str):
        parsed = ast.literal_eval(loc)
        return float(parsed[0]), float(parsed[1])
    raise ValueError(f"Unsupported location format: {type(loc)} -> {loc}")


def calculate_distance_to_goal(x: float, y: float) -> float:
    """
    Distance from shot location to the goal mouth (segment between y=36 and y=44 on x=120).
    This matches your original intention: distance to the closest point on the goal line segment.
    """
    x_dist = GOAL_X - x

    if y < GOALPOST_Y_LOW:
        y_dist = GOALPOST_Y_LOW - y
    elif y > GOALPOST_Y_HIGH:
        y_dist = y - GOALPOST_Y_HIGH
    else:
        y_dist = 0.0

    return math.sqrt(x_dist**2 + y_dist**2)


def calculate_angle_to_goal(x: float, y: float) -> float:
    """
    Opening angle to goalposts, in radians.
    Uses goalposts at (120, 36) and (120, 44).
    """
    a1 = math.atan2(GOALPOST_Y_LOW - y, GOAL_X - x)
    a2 = math.atan2(GOALPOST_Y_HIGH - y, GOAL_X - x)
    angle = abs(a2 - a1)

    # keep within [0, pi]
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle


def add_distance_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: x, y, shot_distance, shot_angle
    Assumes a StatsBomb-like 'location' column.
    """
    out = df.copy()

    # parse location once
    xy = out["location"].apply(parse_location)
    out["x"] = xy.apply(lambda t: t[0])
    out["y"] = xy.apply(lambda t: t[1])

    out["shot_distance"] = out.apply(
        lambda r: calculate_distance_to_goal(r["x"], r["y"]), axis=1)
    out["shot_angle"] = out.apply(
        lambda r: calculate_angle_to_goal(r["x"], r["y"]), axis=1)

    return out


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "train_feat.csv",
    # -----------------------------------------
):
    logger.info(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Generating features...")
    # tqdm is most useful for explicit loops; pandas apply already runs in C/Python.
    # If you want progress bars for apply, consider tqdm.pandas(), but keep it simple first.
    df_feat = add_distance_angle_features(df)

    logger.info(f"Writing features: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(output_path, index=False)

    logger.success("Feature generation complete.")

    # -----------------------------------------


if __name__ == "__main__":
    app()
