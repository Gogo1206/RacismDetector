"""Summarise runs (mean ± std over seeds) and dump test errors for manual analysis.

Run: python -m racism_detector.report
"""

import pandas as pd

from .config import OUTPUT_DIR
from .data import load_splits
from .runs import load_runs, load_test_preds

METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "fpr_race_supportive"]


def label(run: dict) -> str:
    if run["model"] == "tfidf_logreg":
        return "TF-IDF + LogReg"
    if run["aux_weight"] == 0:
        return "DistilRoBERTa single-task"
    return f"DistilRoBERTa multi-task (λ={run['aux_weight']:g})"


def results_table(runs: list[dict]) -> str:
    rows = pd.DataFrame([{"model": label(r), **r["test"]} for r in runs])
    stats = rows.groupby("model", sort=False)[METRICS].agg(["mean", "std"])
    counts = rows.groupby("model", sort=False).size()
    lines = [
        "| Model | runs | " + " | ".join(METRICS) + " |",
        "|---|---|" + "---|" * len(METRICS),
    ]
    for model, row in stats.iterrows():
        cells = []
        for m in METRICS:
            mean, std = row[(m, "mean")], row[(m, "std")]
            cells.append(f"{mean:.3f}" if pd.isna(std) else f"{mean:.3f} ± {std:.3f}")
        lines.append(f"| {model} | {counts[model]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def error_samples(run: dict, n: int = 10) -> str:
    """Most confident false positives and false negatives of one run on test."""
    test = load_splits()["test"].merge(load_test_preds(run["name"]), on="comment_id")
    threshold = run["test"]["threshold"]
    fp = test[(test["label"] == 0) & (test["prob"] >= threshold)].nlargest(n, "prob")
    fn = test[(test["label"] == 1) & (test["prob"] < threshold)].nsmallest(n, "prob")
    out = [f"# Error samples: {run['name']} (threshold {threshold:.3f})", "",
           "Contains offensive text from the dataset.", ""]
    for title, df in (("False positives", fp), ("False negatives", fn)):
        out += [f"## {title}", ""]
        for r in df.itertuples():
            text = r.text.replace("\n", " ")[:300]
            out.append(f"- p={r.prob:.3f} | hate_score={r.hate_speech_score:.2f} | "
                       f"target_race={r.target_race:.2f} | {text}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    # Baseline first, then transformers by aux weight.
    runs = sorted(load_runs(), key=lambda r: (r["model"] != "tfidf_logreg", r["aux_weight"] or 0))
    table = results_table(runs)
    (OUTPUT_DIR / "results.md").write_text("# Test results\n\n" + table + "\n")
    print(table)

    multitask = [r for r in runs if r["aux_weight"]]
    if multitask:
        best = max(multitask, key=lambda r: r["val"]["pr_auc"])
        (OUTPUT_DIR / "error_samples.md").write_text(error_samples(best))
        print(f"\nerror samples from {best['name']} -> outputs/error_samples.md")


if __name__ == "__main__":
    main()
