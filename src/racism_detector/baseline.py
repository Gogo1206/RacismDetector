"""TF-IDF (word + char n-grams) + logistic regression baseline.

Run: python -m racism_detector.baseline
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_union

from .config import SEED
from .data import load_splits, race_supportive_mask
from .metrics import best_threshold, evaluate
from .runs import save_run

C_GRID = (0.25, 1.0, 4.0, 16.0)


def build_features():
    # Char n-grams catch obfuscated spellings that word tokens miss.
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
        ),
    )


def main() -> None:
    splits = load_splits()
    y = {name: df["label"].values for name, df in splits.items()}
    features = build_features()
    X = {"train": features.fit_transform(splits["train"]["text"])}
    for name in ("val", "test"):
        X[name] = features.transform(splits[name]["text"])
    print(f"{X['train'].shape[1]:,} TF-IDF features")

    best_ap, best_c, clf = -1.0, None, None
    for c in C_GRID:
        candidate = LogisticRegression(
            C=c, class_weight="balanced", solver="liblinear", max_iter=2000
        )
        candidate.fit(X["train"], y["train"])
        ap = average_precision_score(y["val"], candidate.predict_proba(X["val"])[:, 1])
        print(f"C={c:<5} val PR-AUC={ap:.3f}")
        if ap > best_ap:
            best_ap, best_c, clf = ap, c, candidate

    probs = {name: clf.predict_proba(X[name])[:, 1] for name in ("val", "test")}
    threshold = best_threshold(y["val"], probs["val"])
    result = {
        "name": "tfidf_logreg",
        "model": "tfidf_logreg",
        "aux_weight": None,
        "seed": SEED,
        "params": {"C": best_c},
        **{
            name: evaluate(
                y[name], probs[name], threshold, race_supportive_mask(splits[name])
            )
            for name in ("val", "test")
        },
    }
    save_run(result, splits["test"], probs["test"])
    print("test:", {k: round(v, 3) for k, v in result["test"].items()})

    # Interpretability: n-grams with the largest positive weights.
    names = features.get_feature_names_out()
    top = np.argsort(clf.coef_[0])[::-1][:20]
    print("top racist-class features:", [names[i].split("__", 1)[1] for i in top])


if __name__ == "__main__":
    main()
