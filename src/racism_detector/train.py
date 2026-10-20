"""Fine-tune DistilRoBERTa. --aux-weight 0 is single-task; > 0 adds the facet heads.

Run: python -m racism_detector.train --aux-weight 1.0 --seed 1
"""

import argparse
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import AUX_TARGETS, MODEL_NAME, MODELS_DIR, SEED
from .data import load_splits, race_supportive_mask
from .metrics import best_threshold, evaluate
from .model import MultiTaskClassifier, multitask_loss
from .runs import save_run


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(df, tokenizer, batch_size, max_length, shuffle, seed=SEED) -> DataLoader:
    texts = df["text"].tolist()
    labels = torch.tensor(df["label"].values, dtype=torch.float32)
    aux = torch.tensor(df[AUX_TARGETS].values, dtype=torch.float32)

    def collate(idx):
        # Pad per batch, not to max_length: most comments are short.
        enc = tokenizer(
            [texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], labels[idx], aux[idx]

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        range(len(df)), batch_size=batch_size, shuffle=shuffle, collate_fn=collate, generator=generator
    )


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for input_ids, mask, _, _ in loader:
        logits, _ = model(input_ids.to(device), mask.to(device))
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(probs).numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--limit", type=int, help="subsample each split (smoke test, nothing saved)")
    p.add_argument("--save-model", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    name = f"distilroberta_aux{args.aux_weight:g}_seed{args.seed}"
    torch.manual_seed(args.seed)
    device = get_device()

    splits = load_splits()
    if args.limit:
        splits = {k: v.sample(min(args.limit, len(v)), random_state=args.seed) for k, v in splits.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    loaders = {
        "train": make_loader(splits["train"], tokenizer, args.batch_size, args.max_length, True, args.seed),
        "val": make_loader(splits["val"], tokenizer, 2 * args.batch_size, args.max_length, False),
        "test": make_loader(splits["test"], tokenizer, 2 * args.batch_size, args.max_length, False),
    }

    model = MultiTaskClassifier.from_pretrained(MODEL_NAME, n_aux=len(AUX_TARGETS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    print(f"{name} on {device}: {len(splits['train'])} train comments, {total_steps} steps")

    best_ap, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, sums = time.time(), np.zeros(3)
        for step, (ids, mask, y, aux_y) in enumerate(loaders["train"], 1):
            main_logits, aux_logits = model(ids.to(device), mask.to(device))
            losses = multitask_loss(main_logits, aux_logits, y.to(device), aux_y.to(device), args.aux_weight)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            sums += [loss.item() for loss in losses]
            if step % 200 == 0:
                total, main, aux = sums / step
                print(f"  epoch {epoch} step {step}: loss {total:.4f} (main {main:.4f}, aux {aux:.4f}) "
                      f"{step / (time.time() - start):.1f} it/s")

        val_ap = average_precision_score(splits["val"]["label"], predict(model, loaders["val"], device))
        total, main, aux = sums / len(loaders["train"])
        history.append({"epoch": epoch, "train_main_loss": main, "train_aux_loss": aux, "val_pr_auc": val_ap})
        print(f"epoch {epoch}: val PR-AUC {val_ap:.4f} ({time.time() - start:.0f}s)")
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Early stopping by val PR-AUC; threshold tuned on val, then applied to test.
    model.load_state_dict(best_state)
    probs = {k: predict(model, loaders[k], device) for k in ("val", "test")}
    threshold = best_threshold(splits["val"]["label"], probs["val"])
    result = {
        "name": name,
        "model": "distilroberta",
        "aux_weight": args.aux_weight,
        "seed": args.seed,
        "params": {**vars(args), "best_epoch": best_epoch},
        "history": history,
        **{
            k: evaluate(splits[k]["label"], probs[k], threshold, race_supportive_mask(splits[k]))
            for k in ("val", "test")
        },
    }
    print("test:", {k: round(v, 3) for k, v in result["test"].items()})
    if args.limit:
        return
    save_run(result, splits["test"], probs["test"])
    if args.save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "model_name": MODEL_NAME,
                "aux_targets": AUX_TARGETS,
                "max_length": args.max_length,
                "threshold": threshold,
            },
            MODELS_DIR / f"{name}.pt",
        )


if __name__ == "__main__":
    main()
