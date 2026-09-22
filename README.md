# NLP Assignment 2 — Language Models and Author Identification

Classifies short text passages as written by J.R.R. Tolkien or Arthur Conan
Doyle, using a from-scratch BPE tokenizer and N-gram language models trained
on *The Hobbit* and *The Lost World*.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```
data/raw/       raw source books (hobbit.txt, lostworld.txt)
data/clean/     cleaned text (Gutenberg boilerplate stripped, lines unwrapped)
src/            implementation
outputs/        trained tokenizer, evaluation results
notes/          assignment instructions
```

## Pipeline

Run in order from the project root (with the venv activated):

1. `python src/clean_data.py` — cleans the raw books into `data/clean/`.
2. `python src/train_final_tokenizer.py` — trains the final BPE tokenizer
   (Whitespace pre-tokenizer, vocab_size=5000, shared across both books) and
   saves it to `outputs/tokenizer_final.json`.
   - `python src/evaluate_tokenizer.py` (optional) — compares vocab
     size / pre-tokenizer combinations; results in
     `outputs/tokenizer_eval_results.csv`.
3. `python src/evaluate_author_models.py` — trains per-author bigram and
   trigram models (90% train / 10% held-out split per book), sweeps add-k
   smoothing values, and evaluates per-passage classification accuracy.
   Results in `outputs/author_model_evaluation.csv`.

## Key implementation notes

- `src/tokenizer.py` wraps Hugging Face's `tokenizers` BPE implementation
  (not a pre-trained tokenizer) — the rest of the project only imports this
  class, never `tokenizers` directly.
- `src/ngram_model.py` implements N-gram counting, add-k smoothing, and
  perplexity from scratch (no NLTK or similar). The same `NGramModel` class
  supports unigram/bigram/trigram via an `order` parameter; unigram counts
  are always available, though the final evaluation compares bigram vs.
  trigram per the assignment's language-modeling requirement.
- Perplexity follows the assignment's own definition exactly: per-word
  average negative log probability, not the exponentiated textbook formula.

## Current best configuration

Bigram model, k=0.1 (add-k smoothing) — selected based on held-out
per-passage classification accuracy (~91% overall). See
`outputs/author_model_evaluation.csv` for the full comparison across
bigram/trigram and k in {0.001, 0.01, 0.1, 1.0, 10.0}.
