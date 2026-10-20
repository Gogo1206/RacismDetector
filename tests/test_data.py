import pandas as pd
import pytest

from racism_detector.config import FACET_MAX
from racism_detector.data import aggregate, race_supportive_mask


def make_raw(rows):
    """Annotator-level rows; facets default to 0."""
    base = {f: 0.0 for f in FACET_MAX}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_aggregate_averages_scaled_facets_and_labels():
    raw = make_raw([
        # Comment 1: hateful and both annotators say it targets race -> racist.
        {"comment_id": 1, "annotator_id": 10, "text": "a", "hate_speech_score": 1.2,
         "target_race": True, "insult": 4.0, "hatespeech": 2.0},
        {"comment_id": 1, "annotator_id": 11, "text": "a", "hate_speech_score": 1.2,
         "target_race": True, "insult": 2.0, "hatespeech": 1.0},
        # Comment 2: hateful but only 1 of 2 annotators says race -> not racist.
        {"comment_id": 2, "annotator_id": 10, "text": "b", "hate_speech_score": 2.0,
         "target_race": True},
        {"comment_id": 2, "annotator_id": 12, "text": "b", "hate_speech_score": 2.0,
         "target_race": False},
        # Comment 3: about race but supportive -> not racist, hard negative.
        {"comment_id": 3, "annotator_id": 11, "text": "c", "hate_speech_score": -2.5,
         "target_race": True},
    ])
    df = aggregate(raw).set_index("comment_id")

    assert df.loc[1, "insult"] == pytest.approx(0.75)  # mean(4/4, 2/4)
    assert df.loc[1, "hatespeech"] == pytest.approx(0.75)  # mean(2/2, 1/2)
    assert df.loc[2, "target_race"] == pytest.approx(0.5)
    assert df["n_annotators"].tolist() == [2, 2, 1]
    assert df["label"].tolist() == [1, 0, 0]
    assert race_supportive_mask(df).tolist() == [False, False, True]
