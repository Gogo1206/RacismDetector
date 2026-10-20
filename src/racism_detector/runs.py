import json

import pandas as pd

from .config import RUNS_DIR


def save_run(result: dict, test_df: pd.DataFrame, test_probs) -> None:
    """Write run metrics as JSON and test-set probabilities as parquet."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    name = result["name"]
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2))
    preds = pd.DataFrame({"comment_id": test_df["comment_id"].values, "prob": test_probs})
    preds.to_parquet(RUNS_DIR / f"{name}_test_preds.parquet", index=False)


def load_runs() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))]


def load_test_preds(name: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS_DIR / f"{name}_test_preds.parquet")
