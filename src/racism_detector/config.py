from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
MODELS_DIR = OUTPUT_DIR / "models"

DATASET_NAME = "ucberkeley-dlab/measuring-hate-speech"
MODEL_NAME = "distilbert/distilroberta-base"
SPLITS = ("train", "val", "test")
SEED = 42

# Ordinal survey items each annotator answered, with their max value.
# Rescaled to [0, 1] and averaged per comment, they become the auxiliary targets.
FACET_MAX = {
    "sentiment": 4,
    "respect": 4,
    "insult": 4,
    "humiliate": 4,
    "status": 4,
    "dehumanize": 4,
    "violence": 4,
    "genocide": 4,
    "attack_defend": 4,
    "hatespeech": 2,
}
# Share of annotators saying the comment targets a racial group.
AUX_TARGETS = [*FACET_MAX, "target_race"]

# hate_speech_score cutoffs suggested by the dataset authors:
# > 0.5 is roughly hate speech, < -1 is counter or supportive speech.
HATE_CUTOFF = 0.5
SUPPORTIVE_CUTOFF = -1.0
# A comment "targets race" when more than half of its annotators said so.
RACE_SHARE_CUTOFF = 0.5
