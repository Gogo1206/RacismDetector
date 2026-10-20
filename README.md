# Racism Detector

Detecting racist online comments with a fine-tuned DistilRoBERTa. The model also learns from the
fine-grained annotator ratings (insult, dehumanization, violence, ...) as auxiliary tasks.

**Question:** a binary "racist?" label throws away most of what annotators recorded. Does
training on those ratings as auxiliary targets make a better classifier than training on the
binary label alone?

## Data

[Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
(Kennedy et al., UC Berkeley D-Lab): 135,556 annotations of 39,565 social media comments.
For every comment, each annotator gave 10 ordinal ratings (sentiment, respect,
insult, humiliate, status, dehumanize, violence, genocide, attack/defend, hate speech) and said
which identity groups it targets. The dataset also provides `hate_speech_score`, a continuous
measure fitted from those ratings with an IRT (Rasch) model.

Annotator rows are collapsed to one row per comment (`src/racism_detector/data.py`):

| Field | Definition |
|---|---|
| `label` (main task) | `hate_speech_score > 0.5` **and** more than half of annotators say it targets race |
| auxiliary targets | each of the 10 ratings rescaled to [0, 1] and averaged over annotators, plus the share of annotators saying it targets race (11 soft targets) |
| race-supportive | targets race but `hate_speech_score < -1` (counter or supportive speech): **hard negatives** |

The 0.5 / -1 cutoffs are the ones the dataset authors suggest. The stratified 80/10/10 split gives
31,652 / 3,956 / 3,957 comments, 6.6% positive in each split, and 459 race-supportive comments in test.

## Models

1. **TF-IDF + logistic regression**: word 1-2-grams + char 3-5-grams (177k features),
   class-balanced, `C` picked on validation.
2. **DistilRoBERTa single-task**: `<s>` token → linear head → racist logit.
3. **DistilRoBERTa multi-task**: same encoder, plus a second linear head that predicts the 11
   auxiliary targets.

```
                 ┌─> main head (1)  ──> BCE(racist label)
text ─> encoder ─┤
                 └─> aux head (11)  ──> BCE(soft facet targets)

loss = BCE_main + λ · BCE_aux        (λ = 0 → single-task, same architecture)
```

Auxiliary labels are used **only during training**. At inference the model reads only text.
The aux head then gives a free explanation: *which* facets drove the score.

Why this might help: the label is a thresholded summary of exactly these ratings, so the aux
targets restore the graded signal that binarization throws away. They also split the label's
AND ("hateful" × "about race") into separate supervised parts.

Training: AdamW, lr 3e-5, batch 32, 3 epochs, 6% linear warmup, max 128 tokens, best epoch picked
by validation PR-AUC. The decision threshold is tuned for F1 on validation, then applied unchanged
to test. Every transformer configuration runs with 3 seeds.

## Results

Test set (3,957 comments, 261 racist). Transformer rows are mean ± std over 3 seeds.
Full table: `outputs/results.md`.

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | FPR on race-supportive |
|---|---|---|---|---|---|---|
| TF-IDF + LogReg | 0.394 | 0.867 | 0.404 | 0.332 | 0.517 | 11.1% |
| DistilRoBERTa single-task | 0.475 ± 0.016 | 0.907 ± 0.009 | 0.470 ± 0.002 | 0.426 ± 0.041 | 0.535 ± 0.073 | 6.8% ± 2.1 |
| **DistilRoBERTa multi-task** (λ=1) | **0.495 ± 0.007** | **0.913 ± 0.005** | **0.498 ± 0.016** | **0.458 ± 0.028** | **0.552 ± 0.072** | **4.8% ± 1.1** |

PR-AUC is the headline metric because only 6.6% of comments are positive. A model that always
says "not racist" gets 93% accuracy and 0.066 PR-AUC.

### Findings

1. **Context beats keywords.** The baseline's highest-weighted features are almost all slurs and
   slur fragments, so it is a keyword matcher. It flags 11.1% of supportive comments about race.
   Fine-tuning cuts that to 6.8% and lifts PR-AUC by 0.08.
2. **Auxiliary facets help, consistently but modestly.** Multi-task improves PR-AUC by 0.020, F1
   by 0.028, and lowers race-supportive false positives from 6.8% to 4.8%. Paired by seed, PR-AUC
   gains are +0.031 / +0.031 / +0.000 and F1 gains are +0.009 / +0.042 / +0.031. On validation,
   the *worst* multi-task epoch (0.504) beat the *best* single-task epoch (0.490), and seed
   variance halved. Caveat: 3 seeds and 261 test positives. The gain is consistent, but this is
   not a significance test.
3. **The aux head is a built-in explanation.** It separates "is this about race?" from "is this
   hostile?":

   | Text | p(racist) | insult | dehumanize | target_race |
   |---|---|---|---|---|
   | "I am a Black woman and proud of my heritage." | 0.002 | 0.14 | 0.10 | **0.91** |
   | "You are an idiot and your code is garbage." | 0.020 | **0.80** | 0.46 | 0.10 |

### Error analysis

Run `python -m racism_detector.report` to write the most confident errors to
`outputs/error_samples.md` (gitignored: it contains offensive dataset text).

- **Missed implicit racism.** Stereotypes and grievance framing with no slur ("X just want special
  treatment", diversity complaints) get near-zero scores.
- **Dialect false positives.** 4 of the 10 most confident false positives are in-group,
  non-hateful uses of a reclaimed slur. This is the African American English bias documented by
  Sap et al. (2019), and the clearest risk of deploying this model.
- **Label noise, not just model error.** 27% of false negatives have a single annotator, vs 15%
  of true positives. Some "racist" labels contain no racial content at all. 30 of 190 false
  positives are hateful comments where some, but not most, annotators marked race, so they
  fall on the label boundary.

### Next steps

Bootstrap confidence intervals on the paired difference; a λ sweep; a dialect audit on
TwitterAAE; cross-dataset testing on HateXplain; length-bucketed batching to speed up training.

## Limitations

- One dataset. Performance on other platforms or newer slang is unmeasured.
- The label depends on cutoff choices (0.5 hate score, >50% race share). Different cutoffs give
  a different task.
- English only. Annotators are US crowd workers, and their judgments carry their own biases.
- MPS training is not bit-for-bit deterministic. Seeds control initialization and data order.
- Intended for **flagging content for human review**, not automatic moderation.

## Project layout

```
src/racism_detector/
  config.py     paths, label cutoffs, aux target list
  data.py       load MHS, aggregate annotators -> comments, label, split
  baseline.py   TF-IDF + logistic regression
  model.py      MultiTaskClassifier + multitask_loss
  train.py      PyTorch fine-tuning loop (CUDA / MPS / CPU)
  metrics.py    PR-AUC, F1, FPR on race-supportive comments, threshold tuning
  report.py     results table (mean ± std over seeds) + error samples
  predict.py    score new text with a saved checkpoint
tests/          aggregation, metrics, model/loss unit tests
scripts/run_all.sh   reproduces every number above
```

## Usage

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest

./scripts/run_all.sh                       # data + baseline + 6 fine-tunes + report (~3 h on M4)

# or step by step
python -m racism_detector.data
python -m racism_detector.baseline
python -m racism_detector.train --aux-weight 1.0 --seed 1 --save-model
python -m racism_detector.report
python -m racism_detector.predict --model outputs/models/distilroberta_aux1_seed1.pt "your text"
```
