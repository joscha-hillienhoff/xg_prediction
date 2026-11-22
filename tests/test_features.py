import math

import pandas as pd
import pytest

from xg_prediction.features import (
    GOAL_X,
    build_features,
    calculate_angle_to_goal,
    calculate_distance_to_goal,
    parse_location,
)
from xg_prediction.modeling.train import BOOL_COLS, preprocessing_pipeline


class TestDistanceToGoal:
    def test_straight_on_from_penalty_spot(self):
        assert calculate_distance_to_goal(108.0, 40.0) == pytest.approx(12.0)

    def test_inside_post_range_ignores_lateral_offset(self):
        # y = 37 is between the posts (36 and 44), so only the horizontal gap counts.
        assert calculate_distance_to_goal(110.0, 37.0) == pytest.approx(10.0)

    def test_outside_post_range_measures_to_nearest_post(self):
        # 20 yards out and 6 yards wide of the near post (y = 36 - 6 = 30): 3-4-5 scaled by 4.
        assert calculate_distance_to_goal(100.0, 30.0) == pytest.approx(math.hypot(20.0, 6.0))

    def test_symmetric_around_goal_centre(self):
        assert calculate_distance_to_goal(100.0, 25.0) == pytest.approx(
            calculate_distance_to_goal(100.0, 55.0)
        )

    def test_shot_from_goal_line_between_posts_is_zero(self):
        assert calculate_distance_to_goal(GOAL_X, 40.0) == 0.0

    def test_shot_from_goal_line_beyond_post_is_lateral_distance(self):
        assert calculate_distance_to_goal(GOAL_X, 30.0) == pytest.approx(6.0)


class TestAngleToGoal:
    def test_penalty_spot(self):
        assert calculate_angle_to_goal(108.0, 40.0) == pytest.approx(2 * math.atan(4 / 12))

    def test_symmetric_around_goal_centre(self):
        assert calculate_angle_to_goal(100.0, 30.0) == pytest.approx(
            calculate_angle_to_goal(100.0, 50.0)
        )

    def test_closer_shots_see_a_wider_goal(self):
        assert calculate_angle_to_goal(110.0, 40.0) > calculate_angle_to_goal(90.0, 40.0)

    def test_shot_from_goal_line_between_posts_sees_the_full_half_plane(self):
        assert calculate_angle_to_goal(GOAL_X, 40.0) == pytest.approx(math.pi)

    def test_shot_from_goal_line_beyond_post_sees_no_goal(self):
        assert calculate_angle_to_goal(GOAL_X, 30.0) == pytest.approx(0.0)

    def test_angle_stays_within_zero_and_pi(self):
        for x in (60.0, 100.0, 119.9, 120.0):
            for y in (0.0, 20.0, 36.0, 40.0, 44.0, 80.0):
                assert 0.0 <= calculate_angle_to_goal(x, y) <= math.pi


class TestParseLocation:
    def test_list(self):
        assert parse_location([100.0, 40.5]) == (100.0, 40.5)

    def test_string_as_read_from_csv(self):
        assert parse_location("[100.0, 40.5]") == (100.0, 40.5)

    def test_unsupported_value_raises(self):
        with pytest.raises(ValueError):
            parse_location(None)


def _raw_shots() -> pd.DataFrame:
    """Two synthetic shots with the identifier and bookkeeping columns of the raw table."""
    freeze_frame = [
        {"location": [118.0, 40.0], "teammate": False, "position": {"name": "Goalkeeper"}},
        {"location": [112.0, 38.0], "teammate": False, "position": {"name": "Center Back"}},
        {"location": [105.0, 45.0], "teammate": True, "position": {"name": "Center Forward"}},
    ]
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "index": [10, 20],
            "timestamp": ["00:10:00.000", "00:20:00.000"],
            "location": ["[108.0, 40.0]", "[100.0, 30.0]"],
            "related_events": ["['a']", "['b']"],
            "shot_key_pass_id": [None, None],
            # As read from the CSV, the freeze frame is the string form of a list.
            "shot_freeze_frame": [repr(freeze_frame)] * 2,
            "player": ["Player A", "Player B"],
            "position": ["Center Forward", "Left Wing"],
            "shot_body_part": ["Right Foot", "Head"],
            "minute": [10, 20],
        }
    )
    for col in BOOL_COLS:
        df[col] = [None, True]
    return df


def test_feature_frame_never_contains_id_or_index():
    features = preprocessing_pipeline(build_features(_raw_shots()))

    assert "id" not in features.columns
    assert "index" not in features.columns
    # Sanity check that the pipeline still produced the engineered columns.
    assert {"shot_distance", "shot_angle", "nearest_defender_dist"} <= set(features.columns)
