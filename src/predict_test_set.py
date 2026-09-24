"""
Final step of the assignment: classify the instructor's test set.

This script adds nothing new to the modelling side -- it reuses the exact
tokenizer and N-gram code the rest of the project already uses:

  - BPETokenizer (tokenizer.py), loaded from outputs/tokenizer_final.json
  - NGramModel and prepare_sequences (ngram_model.py)

What it does, start to finish:

  1. Load the already-trained BPE tokenizer.
  2. Train one bigram model per author on the FULL cleaned text of their
     book. The earlier experiment scripts held out 10% of each book so they
     had something to measure against; here the instructor's test set IS
     the unseen data, so there's no reason to hide any training text.
  3. Read the test file, splitting it into items on each "ITEM-XX" marker.
  4. Score each passage under both authors' models and predict whichever
     model finds it less surprising.
  5. Write outputs/predictions.txt -- one "ITEM-XX<tab>Author" line per
     test item, in the original file order, and nothing else.

The hyperparameters below (order=2, k=0.1) are the ones chosen from
outputs/author_model_evaluation.csv, where that combination gave the best
overall held-out classification accuracy (0.9095).
"""

import re
import sys
from pathlib import Path

from tokenizer import BPETokenizer
from ngram_model import NGramModel, prepare_sequences

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TOKENIZER_PATH = OUTPUTS_DIR / "tokenizer_final.json"

TEST_SET_PATH = PROJECT_ROOT / "data" / "raw" / "HW2-F26-testset.txt"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.txt"

# How many items the instructor's test set is expected to contain. Used
# only as a sanity check at the end -- see run_verification_checks().
EXPECTED_ITEM_COUNT = 48

# Our internal label for each author -> the book that represents them, and
# the exact spelling the submission file has to use.
AUTHOR_FILES = {
    "tolkien": "hobbit.txt",
    "doyle": "lostworld.txt",
}
DISPLAY_NAMES = {
    "tolkien": "Tolkien",
    "doyle": "Doyle",
}

# The final hyperparameter choices.
ORDER = 2          # bigram
SMOOTHING_K = 0.1  # add-k smoothing

# Matches a line that STARTS a new test item: the literal text "ITEM-",
# some digits, then either whitespace or the end of the line. Anything
# captured in group 2 is the start of that item's passage.
ITEM_MARKER_RE = re.compile(r"^(ITEM-\d+)(?:\s+(.*))?$")


def train_author_models(tokenizer: BPETokenizer) -> dict[str, NGramModel]:
    """
    Train one bigram model per author on the complete cleaned text of that
    author's book, using the project's existing NGramModel class.
    Returns a dict: author label -> trained model.
    """
    models = {}
    for author, filename in AUTHOR_FILES.items():
        text = (CLEAN_DIR / filename).read_text(encoding="utf-8")
        sequences = prepare_sequences(text, tokenizer, ORDER)
        model = NGramModel(order=ORDER, vocab_size=tokenizer.vocab_size)
        model.train(sequences)
        models[author] = model
        print(f"trained {author:<8} on all of {filename}: "
              f"{len(sequences)} sentences, {len(model.ngram_counts)} unique bigrams")
    return models


def read_test_items(path: Path) -> list[tuple[str, str]]:
    """
    Read the test file into a list of (item_id, passage_text) pairs, in the
    order they appear in the file.

    A single passage may be wrapped across several physical lines, so we
    can't just treat one line as one item. Instead we walk the file line by
    line: a line beginning with an "ITEM-XX" marker starts a new item, and
    every line after it belongs to that item until the next marker appears.

    Continuation lines are joined with a single space, which is the same
    thing clean_data.py's join_wrapped_lines() does to the training text --
    so test passages and training text are shaped the same way.
    """
    items: list[tuple[str, list[str]]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = ITEM_MARKER_RE.match(line.strip())
        if match:
            # Start a new item. group(2) is the passage text that followed
            # the ID on the same line (None if the ID was alone on its line).
            item_id = match.group(1)
            first_chunk = match.group(2) or ""
            items.append((item_id, [first_chunk] if first_chunk else []))
        elif items and line.strip():
            # A continuation line: append it to the item we're currently in.
            items[-1][1].append(line.strip())
        # A blank line, or text before the very first marker, is ignored.

    return [(item_id, " ".join(chunks).strip()) for item_id, chunks in items]


def classify(text: str, tokenizer: BPETokenizer, models: dict[str, NGramModel]) -> str:
    """
    Predict the author of one passage.

    The passage is split into sentences and encoded with the <s> / </s>
    boundary tokens the bigram model expects (prepare_sequences does both),
    then scored under each author's model using the project's existing
    perplexity method. The author whose model gives the LOWER perplexity --
    i.e. finds the passage less surprising -- is our prediction.

    Both models score the identical sequences, so the per-token averaging
    inside perplexity() divides both sides by the same number; comparing
    the two values is therefore a fair comparison.
    """
    sequences = prepare_sequences(text, tokenizer, ORDER)

    # Safety net: if the sentence splitter found nothing (e.g. a passage
    # with no sentence-ending punctuation), treat the whole passage as one
    # sentence rather than scoring nothing at all.
    if not sequences:
        sequences = [tokenizer.encode_with_boundaries(text, ORDER)]

    best_author = None
    best_perplexity = None
    for author in AUTHOR_FILES:  # fixed order, so any tie breaks the same way every run
        perplexity = models[author].perplexity(sequences, SMOOTHING_K)
        if best_perplexity is None or perplexity < best_perplexity:
            best_perplexity = perplexity
            best_author = author
    return best_author


def run_verification_checks(items: list[tuple[str, str]], path: Path) -> bool:
    """
    Re-read the file we just wrote and check it against the test items.
    Everything here prints to the terminal only -- nothing is written back
    into predictions.txt. Returns True if every check passed.
    """
    written_lines = path.read_text(encoding="utf-8").splitlines()
    input_ids = [item_id for item_id, _ in items]
    output_ids = [line.split("\t")[0] for line in written_lines if "\t" in line]
    labels = [line.split("\t")[1] for line in written_lines if "\t" in line]

    checks = [
        (f"input contains exactly {EXPECTED_ITEM_COUNT} test items",
         len(items) == EXPECTED_ITEM_COUNT),
        (f"predictions.txt contains exactly {EXPECTED_ITEM_COUNT} lines",
         len(written_lines) == EXPECTED_ITEM_COUNT),
        ("every input ID appears exactly once in the output",
         sorted(input_ids) == sorted(output_ids) and len(set(output_ids)) == len(output_ids)),
        ("output order matches input order",
         input_ids == output_ids),
        ("every prediction is exactly 'Doyle' or 'Tolkien'",
         all(label in ("Doyle", "Tolkien") for label in labels)),
        ("each line has exactly one tab",
         all(line.count("\t") == 1 for line in written_lines)),
        ("no blank lines",
         all(line.strip() for line in written_lines)),
    ]

    print("verification:")
    for description, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {description}")
    return all(passed for _, passed in checks)


if __name__ == "__main__":
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
    print(f"loaded tokenizer: vocab_size={tokenizer.vocab_size}")
    models = train_author_models(tokenizer)
    print()

    items = read_test_items(TEST_SET_PATH)
    print(f"read {len(items)} test items from {TEST_SET_PATH.name}")

    # Classify every passage, keeping the file's original item order.
    output_lines = []
    for item_id, text in items:
        prediction = classify(text, tokenizer, models)
        output_lines.append(f"{item_id}\t{DISPLAY_NAMES[prediction]}")

    # Write the submission file: ID, one tab, author name. Nothing else --
    # no header, no scores, no commentary.
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(output_lines)} predictions to {PREDICTIONS_PATH}")
    print()

    all_passed = run_verification_checks(items, PREDICTIONS_PATH)

    tolkien_count = sum(1 for line in output_lines if line.endswith("Tolkien"))
    print()
    print(f"predicted Tolkien: {tolkien_count}, Doyle: {len(output_lines) - tolkien_count}")

    if not all_passed:
        sys.exit("one or more verification checks FAILED -- do not submit this file")
