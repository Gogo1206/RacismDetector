"""Score comments with a saved model. The aux head explains *why* via facet scores.

Run: python -m racism_detector.predict --model outputs/models/<run>.pt "some comment"
"""

import argparse

import torch
from transformers import AutoTokenizer

from .model import MultiTaskClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("texts", nargs="+")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    model = MultiTaskClassifier.from_pretrained(ckpt["model_name"], len(ckpt["aux_targets"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])

    enc = tokenizer(
        args.texts, padding=True, truncation=True, max_length=ckpt["max_length"], return_tensors="pt"
    )
    with torch.no_grad():
        main_logits, aux_logits = model(enc["input_ids"], enc["attention_mask"])
    for text, prob, facets in zip(args.texts, torch.sigmoid(main_logits), torch.sigmoid(aux_logits)):
        verdict = "RACIST" if prob >= ckpt["threshold"] else "not racist"
        print(f"\n{text!r}\n  {verdict}  p={prob:.3f} (threshold {ckpt['threshold']:.3f})")
        print("  " + "  ".join(f"{n}={v:.2f}" for n, v in zip(ckpt["aux_targets"], facets.tolist())))


if __name__ == "__main__":
    main()
