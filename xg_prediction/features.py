import ast
import math
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from xg_prediction.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


GOAL_X = 120.0
GOAL_Y_CENTER = 40.0
GOALPOST_Y_LOW = 36.0
GOALPOST_Y_HIGH = 44.0


def parse_location(loc: object) -> tuple[float, float]:
    """
    Parse a StatsBomb 'location' field into numeric (x, y) coordinates.

    The field may appear either as:
    - a list/tuple [x, y] (already parsed), or
    - a string representation "[x, y]" when loaded from CSV.

    Returns:
        (x, y) as floats in StatsBomb pitch coordinates.
    """
    # Fast path: already a list/tuple
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])

    # CSV case: parse string representation safely
    if isinstance(loc, str):
        parsed = ast.literal_eval(loc)
        return float(parsed[0]), float(parsed[1])

    # Fail fast on unexpected formats to avoid silent feature corruption
    raise ValueError(f"Unsupported location format: {type(loc)} -> {loc}")


def calculate_distance_to_goal(x: float, y: float) -> float:
    """
    Compute the shortest Euclidean distance from the shot location to the goal mouth.

    The goal mouth is modeled as a vertical line segment on the goal line (x = GOAL_X)
    between the two goalposts (y in [GOALPOST_Y_LOW, GOALPOST_Y_HIGH]).

    This definition correctly returns:
    - perpendicular distance for shots aligned with the goal,
    - diagonal distance for shots outside the post range.
    """
    # Horizontal distance to the goal line
    x_dist = GOAL_X - x

    # Vertical distance to the nearest point on the goal mouth segment
    if y < GOALPOST_Y_LOW:
        y_dist = GOALPOST_Y_LOW - y
    elif y > GOALPOST_Y_HIGH:
        y_dist = y - GOALPOST_Y_HIGH
    else:
        # Shot is vertically aligned with the goal mouth
        y_dist = 0.0

    # Euclidean distance to the closest point on the goal mouth
    return math.sqrt(x_dist**2 + y_dist**2)


def calculate_angle_to_goal(x: float, y: float) -> float:
    """
    Compute the opening angle to the goal mouth, in radians.

    The angle is defined as the angle between the two vectors from the shot
    location to the left and right goalposts. A larger angle corresponds to
    a more open goal and typically higher scoring probability.
    """
    # Angle from shot location to the lower goalpost
    a1 = math.atan2(GOALPOST_Y_LOW - y, GOAL_X - x)

    # Angle from shot location to the upper goalpost
    a2 = math.atan2(GOALPOST_Y_HIGH - y, GOAL_X - x)

    # Absolute angular separation between the two rays
    angle = abs(a2 - a1)

    # Numerical safeguard: ensure angle lies within [0, pi]
    # (handles rare wrap-around cases)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle


def add_distance_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic geometric shot features derived from the shot location.

    Specifically adds:
      - x, y: numeric pitch coordinates
      - shot_distance: distance to the goal mouth
      - shot_angle: opening angle to the goal
    """
    # Work on a copy to avoid mutating the original DataFrame
    out = df.copy()

    # Parse the StatsBomb 'location' field once to avoid repeated parsing
    # in downstream feature calculations
    xy = out["location"].apply(parse_location)

    # Split coordinates into explicit numeric columns
    out["x"] = xy.apply(lambda t: t[0])
    out["y"] = xy.apply(lambda t: t[1])

    # Distance to the closest point on the goal mouth
    out["shot_distance"] = out.apply(
        lambda r: calculate_distance_to_goal(r["x"], r["y"]),
        axis=1,
    )

    # Opening angle to the goalposts
    out["shot_angle"] = out.apply(
        lambda r: calculate_angle_to_goal(r["x"], r["y"]),
        axis=1,
    )

    return out


def parse_freeze_frame(value):
    """
    Parse the StatsBomb 'shot_freeze_frame' field into a Python list.

    The column may contain:
    - None / NaN / empty strings (no freeze-frame recorded)
    - an already-parsed list (e.g. from JSON input)
    - a string representation of a list when loaded from CSV

    Returns an empty list if no valid freeze-frame information is present.
    """
    # Handle missing or empty values explicitly to avoid downstream checks
    if value is None or value == "" or value == "[]" or pd.isna(value):
        return []

    # Fast path: already parsed
    if isinstance(value, list):
        return value

    # CSV case: safely parse string representation into Python objects
    return ast.literal_eval(value)


def split_freeze_frame(freeze_frame):
    """
    Split a freeze-frame snapshot into defenders, teammates, and goalkeeper.

    Returns:
        defenders  : list of opponent outfield players
        teammates  : list of attacking teammates (excluding the shooter)
        goalkeeper : opponent goalkeeper dict (or None if missing)
    """
    defenders = []
    teammates = []
    goalkeeper = None

    # Ensure freeze_frame is always a list, regardless of input format
    freeze_frame = parse_freeze_frame(freeze_frame)

    for p in freeze_frame:
        if p["teammate"]:
            # Attacking teammates (useful for density / passing context)
            teammates.append(p)
        else:
            # Opponent players: distinguish GK from outfield defenders
            if p["position"]["name"] == "Goalkeeper":
                goalkeeper = p
            else:
                defenders.append(p)

    return defenders, teammates, goalkeeper


def nearest_defender_distance(defenders, shot_x, shot_y):
    """
    Compute the distance from the shot location to the closest opponent defender.

    Returns None if no defenders are present, allowing the caller to decide
    how to handle missing defensive pressure (e.g., open-goal situations).
    """
    # No defenders recorded in the freeze-frame
    if not defenders:
        return None

    # Minimum Euclidean distance from the shooter to any defender
    return min(math.dist((shot_x, shot_y), tuple(d["location"])) for d in defenders)


def add_nearest_defender_feature(df):
    """
    Add a feature representing the distance to the nearest opponent defender.

    The feature is derived from the StatsBomb freeze-frame and captures
    immediate defensive pressure on the shooter.
    """
    # Work on a copy to avoid mutating the original DataFrame
    out = df.copy()

    def compute(row):
        # Extract freeze-frame snapshot for the current shot
        freeze_frame = row["shot_freeze_frame"]

        # No freeze-frame available (e.g., tracking not recorded)
        # Use NaN so downstream models can handle missing values explicitly
        if not freeze_frame:
            return np.nan

        # Split freeze-frame players into semantic groups
        defenders, _, _ = split_freeze_frame(freeze_frame)

        # Freeze-frame present but no opponent defenders (rare, e.g. open goal)
        if not defenders:
            return np.nan

        # Compute nearest defender distance using shot coordinates
        return nearest_defender_distance(defenders, row["x"], row["y"])

    # Apply row-wise due to variable-length freeze-frame data
    out["nearest_defender_dist"] = out.apply(compute, axis=1)

    return out


def defenders_in_lane(defenders, shot_x, shot_y, width=2.5):
    """
    Count the number of opponent defenders positioned in the shot lane.

    The shot lane is approximated as a corridor of fixed width around the
    line segment from the shot location to the center of the goal.
    """
    # Goal center in StatsBomb coordinates
    gx, gy = 120.0, 40.0
    count = 0

    for d in defenders:
        dx, dy = d["location"]

        # Compute projection parameter t of the defender onto the shot→goal line
        # t < 0   → behind the shooter
        # t > 1   → beyond the goal
        num = (dx - shot_x) * (gx - shot_x) + (dy - shot_y) * (gy - shot_y)
        den = (gx - shot_x) ** 2 + (gy - shot_y) ** 2
        t = num / den if den > 0 else -1

        # Only consider defenders located between shooter and goal
        if 0 < t < 1:
            # Coordinates of the perpendicular projection onto the shot line
            px = shot_x + t * (gx - shot_x)
            py = shot_y + t * (gy - shot_y)

            # Perpendicular distance from defender to the shot line
            dist = math.dist((dx, dy), (px, py))

            # Defender blocks the lane if within the corridor width
            if dist < width:
                count += 1

    return count


def add_defenders_in_lane_feature(df):
    """
    Add a feature counting the number of defenders blocking the shot lane.

    This feature captures how congested the shooting corridor is and serves
    as a proxy for block probability and shooting difficulty.
    """
    # Work on a copy to avoid mutating the original DataFrame
    out = df.copy()

    def compute(row):
        # Extract freeze-frame snapshot for the current shot
        freeze_frame = row["shot_freeze_frame"]

        # No freeze-frame available → missing value
        if not freeze_frame:
            return np.nan

        # Extract opponent defenders from the freeze-frame
        defenders, _, _ = split_freeze_frame(freeze_frame)

        # Freeze-frame present but no defenders (e.g., open-goal situations)
        if not defenders:
            return np.nan

        # Count defenders positioned in the shot lane
        return defenders_in_lane(defenders, row["x"], row["y"])

    # Row-wise apply is required due to variable-length freeze-frame data
    out["defenders_in_lane"] = out.apply(compute, axis=1)

    return out


def goalkeeper_distance(gk, shot_x, shot_y):
    """
    Compute the distance from the shot location to the opponent goalkeeper.

    Returns None if no goalkeeper is present in the freeze-frame.
    """
    # No goalkeeper recorded (rare, but possible in tracking data)
    if gk is None:
        return None

    # Euclidean distance between shooter and goalkeeper
    return math.dist((shot_x, shot_y), tuple(gk["location"]))


def add_goalkeeper_distance_feature(df):
    """
    Add a feature representing the distance between the shooter and the opponent goalkeeper.

    This feature captures goalkeeper positioning in 1v1 situations and helps
    distinguish close-range chances with an advanced goalkeeper from deeper shots.
    """
    # Work on a copy to avoid mutating the original DataFrame
    out = df.copy()

    def compute(row):
        # Extract freeze-frame snapshot for the current shot
        freeze_frame = row["shot_freeze_frame"]

        # No freeze-frame available (tracking not recorded)
        if not freeze_frame:
            return np.nan

        # Extract goalkeeper from the freeze-frame
        _, _, goalkeeper = split_freeze_frame(freeze_frame)

        # Freeze-frame present but goalkeeper missing (rare edge case)
        if not goalkeeper:
            return np.nan

        # Compute Euclidean distance between shooter and goalkeeper
        return goalkeeper_distance(goalkeeper, row["x"], row["y"])

    # Row-wise apply required due to variable-length freeze-frame data
    out["goalkeeper_distance"] = out.apply(compute, axis=1)

    return out


def goalkeeper_off_center(gk):
    """
    Compute the goalkeeper's lateral displacement from the goal center.

    A larger value indicates that the goalkeeper is positioned away from the
    center of the goal, potentially leaving the far post exposed.
    """
    # No goalkeeper recorded
    if gk is None:
        return None

    # Extract goalkeeper y-coordinate and measure deviation from goal center (y = 40)
    _, gk_y = gk["location"]
    return abs(gk_y - 40.0)


def add_goalkeeper_off_center_feature(df):
    """
    Add a feature measuring how far the goalkeeper is positioned from the goal center.

    This feature complements shot angle by capturing open-goal or far-post situations
    that angle alone cannot fully explain.
    """
    # Work on a copy to avoid mutating the original DataFrame
    out = df.copy()

    def compute(row):
        # Extract freeze-frame snapshot for the current shot
        freeze_frame = row["shot_freeze_frame"]

        # No freeze-frame available → missing value
        if not freeze_frame:
            return np.nan

        # Extract goalkeeper from the freeze-frame
        _, _, goalkeeper = split_freeze_frame(freeze_frame)

        # Freeze-frame present but goalkeeper missing
        if not goalkeeper:
            return np.nan

        # Compute lateral displacement from goal center
        return goalkeeper_off_center(goalkeeper)

    # Row-wise apply due to per-shot freeze-frame variability
    out["goalkeeper_off_center"] = out.apply(compute, axis=1)

    return out


def defenders_within_radius(defenders, shot_x, shot_y, radius=5.0):
    """
    Counts how many defenders are within a given Euclidean distance
    of the shot location.

    Parameters
    ----------
    defenders : list of dict
        Defender objects with a 'location' field [x, y]
    shot_x, shot_y : float
        Shot location coordinates
    radius : float, optional
        Radius around the shot within which defenders are counted

    Returns
    -------
    int
        Number of defenders within the given radius
    """

    # Count defenders whose distance to the shot location
    # is less than or equal to the specified radius
    return sum(math.dist((shot_x, shot_y), tuple(d["location"])) <= radius for d in defenders)


def add_defenders_in_radius_feature(df):
    """
    Adds a feature counting the number of defenders close to the shot location.

    For shots without freeze-frame information (e.g. penalties),
    the feature is set to NaN because defender proximity is not defined.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing shot and freeze-frame data

    Returns
    -------
    pandas.DataFrame
        Dataframe with added 'defenders_in_radius' feature
    """

    out = df.copy()

    def compute(row):
        freeze_frame = row["shot_freeze_frame"]

        # Missing or empty freeze frame (e.g. penalties)
        # → defender proximity cannot be computed
        if not freeze_frame:
            return np.nan

        # Split freeze frame into defenders, teammates, goalkeeper
        defenders, _, _ = split_freeze_frame(freeze_frame)

        # No defenders present → feature undefined
        if not defenders:
            return np.nan

        # Count defenders within the specified radius of the shot
        return defenders_within_radius(defenders, row["x"], row["y"])

    # Apply row-wise feature computation
    out["defenders_in_radius"] = out.apply(compute, axis=1)

    return out


#  Ordered list of feature engineering steps.
FEATURE_STEPS = [
    add_distance_angle_features,
    add_nearest_defender_feature,
    add_defenders_in_lane_feature,
    add_goalkeeper_distance_feature,
    add_goalkeeper_off_center_feature,
    add_defenders_in_radius_feature,
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in a deterministic, ordered pipeline.

    This function serves as the single entry point for feature creation and is
    used identically during training and inference to guarantee consistency.
    """
    # Work on a copy to avoid mutating the original input DataFrame
    out = df.copy()

    # Sequentially apply each feature step; each step is responsible
    # for adding its own columns and preserving existing ones.
    for step in FEATURE_STEPS:
        out = step(out)

    return out


@app.command()
def main(
    # ---- Input and Output Path ----
    input_train: Path = RAW_DATA_DIR / "train.csv",
    input_test: Path = RAW_DATA_DIR / "test.csv",
    output_train: Path = PROCESSED_DATA_DIR / "train_feat.csv",
    output_test: Path = PROCESSED_DATA_DIR / "test_feat.csv"
    # -----------------------------------------
):
    logger.info(f"Loading dataset: {input_train} and {input_test}")
    df_train = pd.read_csv(input_train)
    df_test = pd.read_csv(input_test)

    logger.info("Generating features...")
    df_train_feat = build_features(df_train)
    df_test_feat = build_features(df_test)

    logger.info(f"Writing features: {output_train} and {output_test}")
    output_train.parent.mkdir(parents=True, exist_ok=True)
    output_test.parent.mkdir(parents=True, exist_ok=True)
    
    df_train_feat.to_csv(output_train, index=False)
    df_test_feat.to_csv(output_test, index=False)

    logger.success("Feature generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
