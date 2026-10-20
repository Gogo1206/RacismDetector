"""Load Measuring Hate Speech, collapse annotator rows to one row per comment, split.

Run: python -m racism_detector.data
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from .config import (
    DATASET_NAME,
    FACET_MAX,
    HATE_CUTOFF,
    PROCESSED_DIR,
    RACE_SHARE_CUTOFF,
    SEED,
    SPLITS,
    SUPPORTIVE_CUTOFF,
)


def load_raw() -> pd.DataFrame:
    return load_dataset(DATASET_NAME, split="train").to_pandas()


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per comment: text, soft auxiliary targets in [0, 1], binary label."""
    scaled = raw[list(FACET_MAX)] / pd.Series(FACET_MAX)
    scaled["target_race"] = raw["target_race"].astype(float)
    scaled["comment_id"] = raw["comment_id"]
    aux = scaled.groupby("comment_id").mean()

    meta = raw.groupby("comment_id").agg(
        text=("text", "first"),
        hate_speech_score=("hate_speech_score", "first"),
        n_annotators=("annotator_id", "size"),
    )
    df = meta.join(aux).reset_index()
    is_hate = df["hate_speech_score"] > HATE_CUTOFF
    targets_race = df["target_race"] > RACE_SHARE_CUTOFF
    df["label"] = (is_hate & targets_race).astype(int)
    return df


def race_supportive_mask(df: pd.DataFrame) -> pd.Series:
    """Comments about race that are supportive or counter-speech: the hard negatives."""
    return (df["target_race"] > RACE_SHARE_CUTOFF) & (
        df["hate_speech_score"] < SUPPORTIVE_CUTOFF
    )


def split(df: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Stratified 80/10/10 split."""
    train, rest = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label"], random_state=seed
    )
    return {"train": train, "val": val, "test": test}


def load_splits() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in SPLITS}


def main() -> None:
    df = aggregate(load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in split(df).items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        print(
            f"{name:5s}: {len(part):6d} comments | {part['label'].mean():.1%} racist"
            f" | {race_supportive_mask(part).sum()} race-supportive"
        )


if __name__ == "__main__":
    main()
