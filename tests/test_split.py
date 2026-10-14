import pandas as pd
import pytest

from recsys_auto.split import temporal_fold


def test_equal_timestamps_stay_in_one_period():
    events = pd.DataFrame({"timestamp": [1, 1, 2, 2, 2, 3, 4, 4, 5, 6]})
    fold = temporal_fold(events, [0.4, 0.7, 1.0])
    assert fold.train_end == 2
    assert fold.validation_end == 6
    for left, right in [(fold.train, fold.validation), (fold.validation, fold.test)]:
        assert events.iloc[left].timestamp.max() < events.iloc[right].timestamp.min()


def test_unsorted_events_rejected():
    with pytest.raises(ValueError):
        temporal_fold(pd.DataFrame({"timestamp": [2, 1, 3, 4]}), [0.3, 0.6, 1.0])
