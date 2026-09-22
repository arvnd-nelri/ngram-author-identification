"""
Single evaluation workflow for the N-gram / author-identification part of
the assignment: train per-author bigram/trigram models on a 90% training
split of each book, then evaluate every (order, k) combination on the
held-out 10% using ONE method.

Unigram counts are still collected wherever NGramModel is used elsewhere
(order=1 is fully supported in ngram_model.py, per the assignment's count-
collection requirement) -- this script just doesn't evaluate a unigram
language model, since the assignment's language-modeling/perplexity
comparison is specifically bigram vs. trigram.

  - average perplexity of each author's held-out text under its own model
    (assignment.md's exact definition: per-word average negative log
    probability, NOT exponentiated -- see NGramModel.perplexity())
  - per-passage classification accuracy: for each individual held-out
    sentence, predict whichever author's model gives it the lower
    perplexity, and check that against the true author

Everything lands in one table, saved to outputs/author_model_evaluation.csv.

The instructor's official test set is not available yet and is out of
scope for this script -- it only uses each book's own held-out 10% split.
"""

from pathlib import Path

import pandas as pd

from tokenizer import BPETokenizer
from ngram_model import NGramModel, prepare_sequences

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TOKENIZER_PATH = OUTPUTS_DIR / "tokenizer_final.json"

# Maps our label for each author to the book file that represents them.
AUTHOR_FILES = {
    "tolkien": "hobbit.txt",
    "doyle": "lostworld.txt",
}

ORDERS = [2, 3]  # bigram, trigram -- unigram counts are still supported in NGramModel, just not evaluated here
K_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
HOLD_OUT_FRACTION = 0.1


def split_train_and_eval(text: str) -> tuple[str, str]:
    """
    Split a book's cleaned text into a training portion and a held-out
    portion, by paragraph (paragraphs are separated by a blank line). The
    last HOLD_OUT_FRACTION of paragraphs become the held-out text.
    """
    paragraphs = text.split("\n\n")
    split_point = int(len(paragraphs) * (1 - HOLD_OUT_FRACTION))
    train_paragraphs = paragraphs[:split_point]
    eval_paragraphs = paragraphs[split_point:]
    return "\n\n".join(train_paragraphs), "\n\n".join(eval_paragraphs)


def classify_sequence(sequence: list[int], models_for_order: dict[str, NGramModel], k: float) -> str:
    """
    Predict the author of one held-out sentence: score it (as a
    single-sentence "test set") under each author's model using the
    assignment's perplexity definition, and return whichever author's
    model gives the lower value.
    """
    best_author = None
    best_perplexity = None
    for author, model in models_for_order.items():
        pp = model.perplexity([sequence], k)
        if best_perplexity is None or pp < best_perplexity:
            best_perplexity = pp
            best_author = author
    return best_author


if __name__ == "__main__":
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)

    # Read each author's book and split it into train / held-out portions.
    train_text = {}
    eval_text = {}
    for author, filename in AUTHOR_FILES.items():
        full_text = (CLEAN_DIR / filename).read_text(encoding="utf-8")
        train_text[author], eval_text[author] = split_train_and_eval(full_text)

    # Train one NGramModel per (author, order) -- 6 models total. Counts
    # don't depend on k, so this only needs to happen once.
    models: dict[str, dict[int, NGramModel]] = {}
    eval_sequences: dict[str, dict[int, list[list[int]]]] = {}
    for author in AUTHOR_FILES:
        models[author] = {}
        eval_sequences[author] = {}
        for order in ORDERS:
            train_sequences = prepare_sequences(train_text[author], tokenizer, order)
            model = NGramModel(order=order, vocab_size=tokenizer.vocab_size)
            model.train(train_sequences)
            models[author][order] = model
            eval_sequences[author][order] = prepare_sequences(eval_text[author], tokenizer, order)
            print(f"trained {author} order={order}: "
                  f"{len(train_sequences)} training sentences, "
                  f"{len(eval_sequences[author][order])} held-out sentences")
    print()

    # For every (order, k) combination, compute perplexity (for comparing
    # unigram/bigram/trigram and different k, per the assignment) and
    # per-passage classification accuracy (for author identification).
    results = []
    for order in ORDERS:
        models_for_order = {author: models[author][order] for author in AUTHOR_FILES}
        for k in K_VALUES:
            row = {"order": order, "k": k}

            accuracy_by_author = {}
            for author in AUTHOR_FILES:
                sequences = eval_sequences[author][order]

                # Average perplexity of this author's held-out text under
                # its OWN model -- the assignment's exact definition,
                # used only for comparing models, not for classification.
                row[f"avg_perplexity_{author}"] = round(models_for_order[author].perplexity(sequences, k), 4)

                # Per-passage classification accuracy.
                predictions = [classify_sequence(seq, models_for_order, k) for seq in sequences]
                num_correct = sum(1 for pred in predictions if pred == author)
                accuracy_by_author[author] = num_correct / len(predictions)

            row["tolkien_accuracy"] = round(accuracy_by_author["tolkien"], 4)
            row["doyle_accuracy"] = round(accuracy_by_author["doyle"], 4)

            total_correct = sum(
                accuracy_by_author[author] * len(eval_sequences[author][order])
                for author in AUTHOR_FILES
            )
            total_sentences = sum(len(eval_sequences[author][order]) for author in AUTHOR_FILES)
            row["overall_accuracy"] = round(total_correct / total_sentences, 4)

            results.append(row)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUTS_DIR / "author_model_evaluation.csv"
    results_df.to_csv(csv_path, index=False)
    print()
    print(f"saved results table to {csv_path}")

    print()
    print("best (order, k) by overall_accuracy:")
    best_row = results_df.loc[results_df["overall_accuracy"].idxmax()]
    print(best_row.to_string())
