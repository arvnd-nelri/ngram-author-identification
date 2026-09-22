"""
Train and save the final BPE tokenizer used for the rest of this project.

The hyperparameters below (pre_tokenizer="whitespace", vocab_size=5000) were
chosen by comparing a grid of configurations in evaluate_tokenizer.py -- see
outputs/tokenizer_eval_results.csv for the comparison data. whitespace/5000
gave the best fragmentation behaviour on unseen vocabulary while keeping
the vocabulary small enough to limit N-gram sparsity later, at only a small
compression cost versus larger vocab sizes.

Unlike evaluate_tokenizer.py (which trains on a 90% split so 10% can be
held out for measuring quality), this script trains on the FULL cleaned
text of both books -- once the config is decided, there's no reason to
hold data back from the tokenizer actually used downstream.
"""

from pathlib import Path

from tokenizer import BPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

VOCAB_SIZE = 5000
PRE_TOKENIZER = "whitespace"
MIN_FREQUENCY = 2

BOOK_FILENAMES = ["hobbit.txt", "lostworld.txt"]


if __name__ == "__main__":
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, pre_tokenizer=PRE_TOKENIZER, min_frequency=MIN_FREQUENCY)
    tokenizer.train([CLEAN_DIR / filename for filename in BOOK_FILENAMES])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUTS_DIR / "tokenizer_final.json"
    tokenizer.save(save_path)

    print(f"trained on full text of: {', '.join(BOOK_FILENAMES)}")
    print(f"pre_tokenizer: {PRE_TOKENIZER}, requested vocab_size: {VOCAB_SIZE}")
    print(f"actual vocab_size: {tokenizer.vocab_size}")
    print(f"saved to: {save_path}")
