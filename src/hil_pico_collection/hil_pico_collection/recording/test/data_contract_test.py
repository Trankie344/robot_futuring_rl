import pandas as pd

from hil_pico_collection.adapters.base import CANONICAL_NAMES
from hil_pico_collection.recording.v21_writer import FEATURES, _expert_frame_segments


def test_state_and_action_names_match_rl_token_canonical_order():
    assert FEATURES["observation.state"]["names"] == list(CANONICAL_NAMES)
    assert FEATURES["action"]["names"] == list(CANONICAL_NAMES)
    assert FEATURES["observation.state"]["dtype"] == "float32"
    assert FEATURES["action"]["shape"] == [16]


def test_stage2_required_provenance_columns_have_exact_types():
    assert FEATURES["intervention"] == {"dtype": "bool", "shape": [1], "names": None}
    assert FEATURES["control_mode"] == {"dtype": "int64", "shape": [1], "names": None}


def test_expert_segments_use_only_explicit_intervention():
    frame = pd.DataFrame(
        {
            "frame_index": [0, 1, 2, 3, 4],
            "control_mode": [5, 5, 1, 5, 1],
            "intervention": [False, True, True, False, False],
        }
    )
    assert _expert_frame_segments(frame) == [{"start_frame_index": 1, "end_frame_index": 2}]


def test_teleoperation_mode_without_intervention_is_not_expert():
    frame = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "control_mode": [5, 5],
            "intervention": [False, False],
        }
    )
    assert _expert_frame_segments(frame) == []
