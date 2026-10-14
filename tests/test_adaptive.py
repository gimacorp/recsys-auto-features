import pandas as pd

from recsys_auto.adaptive import propose_windows


def test_window_proposals_use_timing_independently_of_labels():
    training = pd.DataFrame(
        {
            "event_id": range(6),
            "timestamp": [0, 86400, 2 * 86400, 6 * 86400, 14 * 86400, 40 * 86400],
            "user": [1] * 6,
            "item": [1] * 6,
            "genres": [(1,)] * 6,
            "rating": [5] * 6,
            "label": [1] * 6,
        }
    )
    first, details = propose_windows(training, [7, 30, 90, None])
    training["label"] = 0
    training["rating"] = 1
    second, _ = propose_windows(training, [7, 30, 90, None])
    assert first == second
    assert details["user"]["days"]
    assert all(pack.days not in [7, 30, 90] for pack in first)
