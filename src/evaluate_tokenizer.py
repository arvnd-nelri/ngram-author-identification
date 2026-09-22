"""
Compare BPE tokenizer configurations (vocab_size x pre_tokenizer) so we can
pick one with our eyes open, instead of guessing from a single sentence.

Why not just eyeball one sentence: a short sentence made of common words
(like "In a hole in the ground there lived a hobbit.") looks fine under
almost ANY config, because BPE always learns frequent words as whole tokens
first, regardless of vocab size. The differences between configs only show
up on (a) how much LESS common words get fragmented, and (b) aggregate
behaviour over a large chunk of held-out text. So this script measures both.

This file only imports BPETokenizer from tokenizer.py -- it does not touch
the `tokenizers` library directly. All core tokenizer logic stays in
tokenizer.py; this file is purely evaluation/reporting.
"""

import tempfile
from pathlib import Path

import pandas as pd

from tokenizer import BPETokenizer, PRE_TOKENIZER_CHOICES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BOOK_FILENAMES = ["hobbit.txt", "lostworld.txt"]

# The grid of configurations to compare.
VOCAB_SIZES = [1000, 2000, 5000, 8000]
PRE_TOKENIZERS = list(PRE_TOKENIZER_CHOICES)  # ["whitespace", "whitespace_split", "byte_level"]

# Held-out fraction: the LAST this-fraction of each book's paragraphs is
# excluded from training and used only to measure tokenizer quality on
# text the tokenizer has not memorized.
HOLD_OUT_FRACTION = 0.1

# Words used to check how well each config preserves AUTHOR-DISTINCTIVE
# vocabulary -- exactly the kind of word an N-gram classifier would need
# to tell the two authors apart. "Baggins" is Tolkien-specific, "Challenger"
# is Conan Doyle-specific (Professor Challenger), and the invented word is
# a stand-in for any word neither author's vocabulary anticipated.
SAMPLE_WORDS = ["Baggins", "Challenger", "flibbertigibbet"]


def split_train_and_eval(text: str) -> tuple[str, str]:
    """
    Split cleaned book text into a training portion and a held-out
    evaluation portion, by paragraph (paragraphs are separated by a blank
    line, per how clean_data.py wrote this text). The last HOLD_OUT_FRACTION
    of paragraphs become the eval text; the rest is training text.
    """
    paragraphs = text.split("\n\n")
    split_point = int(len(paragraphs) * (1 - HOLD_OUT_FRACTION))
    train_paragraphs = paragraphs[:split_point]
    eval_paragraphs = paragraphs[split_point:]
    return "\n\n".join(train_paragraphs), "\n\n".join(eval_paragraphs)


def load_train_and_eval_texts() -> tuple[dict[str, str], str]:
    """
    Read both cleaned books and split each into train/eval text.
    Returns (train_texts, combined_eval_text), where train_texts maps
    filename -> training text (kept separate per book, since BPETokenizer
    trains from a list of files), and combined_eval_text is both books'
    held-out portions joined into one string (evaluation doesn't need to
    know which book a paragraph came from).
    """
    train_texts = {}
    eval_texts = []

    for filename in BOOK_FILENAMES:
        full_text = (CLEAN_DIR / filename).read_text(encoding="utf-8")
        train_text, eval_text = split_train_and_eval(full_text)
        train_texts[filename] = train_text
        eval_texts.append(eval_text)

    combined_eval_text = "\n\n".join(eval_texts)
    return train_texts, combined_eval_text


def evaluate_config(vocab_size: int, pre_tokenizer: str, train_texts: dict[str, str], eval_text: str) -> dict:
    """
    Train one BPETokenizer config on the training text and measure its
    quality on the held-out eval text. Returns a dict of results (one row
    of the final table).
    """
    # BPETokenizer.train() takes file paths, so write the training text to
    # temporary files. The temp directory is deleted automatically once
    # we leave the `with` block, so no leftover files in the project.
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_file_paths = []
        for filename, text in train_texts.items():
            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_text(text, encoding="utf-8")
            train_file_paths.append(tmp_path)

        tokenizer = BPETokenizer(vocab_size=vocab_size, pre_tokenizer=pre_tokenizer, min_frequency=2)
        tokenizer.train(train_file_paths)

    # --- Metric 1 & 2: tokens-per-word and UNK rate on held-out text ---
    eval_word_count = len(eval_text.split())
    eval_ids = tokenizer.encode(eval_text)
    eval_token_count = len(eval_ids)
    unk_count = sum(1 for token_id in eval_ids if token_id == tokenizer.unk_id)

    tokens_per_word = eval_token_count / eval_word_count
    unk_rate = unk_count / eval_token_count

    # --- Metric 3: fragmentation of author-distinctive sample words ---
    row = {
        "pre_tokenizer": pre_tokenizer,
        "vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "tokens_per_word": round(tokens_per_word, 3),
        "unk_rate_pct": round(unk_rate * 100, 3),
    }
    for word in SAMPLE_WORDS:
        num_subtokens = len(tokenizer.encode_tokens(word))
        row[f"'{word}'_subtokens"] = num_subtokens

    return row


if __name__ == "__main__":
    train_texts, eval_text = load_train_and_eval_texts()
    print(f"held-out eval text: {len(eval_text.split())} words "
          f"({HOLD_OUT_FRACTION:.0%} of each book, by paragraph)")
    print(f"testing {len(PRE_TOKENIZERS)} pre-tokenizers x {len(VOCAB_SIZES)} vocab sizes "
          f"= {len(PRE_TOKENIZERS) * len(VOCAB_SIZES)} configs...")
    print()

    results = []
    for pre_tokenizer in PRE_TOKENIZERS:
        for vocab_size in VOCAB_SIZES:
            row = evaluate_config(vocab_size, pre_tokenizer, train_texts, eval_text)
            results.append(row)
            print(f"done: pre_tokenizer={pre_tokenizer:<17} vocab_size={vocab_size}")

    results_df = pd.DataFrame(results)

    print()
    print(results_df.to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_DIR / "tokenizer_eval_results.csv"
    results_df.to_csv(csv_path, index=False)
    print()
    print(f"saved results table to {csv_path}")
