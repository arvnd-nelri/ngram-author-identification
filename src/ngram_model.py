"""
From-scratch N-gram language modeling: sentence splitting, count collection,
add-k smoothed probability, and perplexity.

The same NGramModel class is used for unigram (order=1), bigram (order=2),
and trigram (order=3) models -- the counting and probability math is
identical, only the window size changes.

No N-gram-modeling library is used here (no NLTK, etc.) -- counts are
plain dictionaries and probabilities are computed by hand.
"""

import math
import re
from collections import defaultdict

from tokenizer import BPETokenizer

# Common abbreviations that end in a period but do NOT mark a sentence end
# (e.g. "Mr. Baggins" should not be split into two sentences at "Mr.").
# This is a small safety net, not an exhaustive abbreviation list -- some
# unusual abbreviations will still cause an incorrect split, which is an
# acceptable tradeoff for a "reasonably robust, not over-engineered" splitter.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr",
    "vs", "etc", "mt", "capt", "col", "gen", "lt", "rev",
}

# Split the text right after a sentence-ending punctuation mark (., !, or ?),
# optionally followed by one closing quote/bracket character, wherever
# whitespace follows. Requiring trailing whitespace means "..." (no space
# between the dots) is never split mid-ellipsis. A blank line between
# paragraphs is just another run of whitespace, so paragraph breaks are
# naturally treated as sentence breaks too, with no extra handling needed.
_SPLIT_RE = re.compile(r'(?<=[.!?])(?:["\')\]])?\s+')

# Grabs the last word-like chunk right before the end of a fragment, so we
# can check whether it looks like an abbreviation or a single initial.
_LAST_WORD_RE = re.compile(r"([A-Za-z]+)[.!?]?[\"')\]]?$")


def _ends_with_abbreviation(fragment: str) -> bool:
    """
    Heuristic: does this fragment end right after something that looks like
    an abbreviation (e.g. "Mr.") or a single initial (e.g. "J." as in
    "J. R. R. Tolkien"), rather than a genuine sentence end?
    """
    match = _LAST_WORD_RE.search(fragment)
    if not match:
        return False
    word = match.group(1)
    if len(word) == 1:
        return True
    return word.lower() in ABBREVIATIONS


def split_into_sentences(text: str) -> list[str]:
    """
    Split cleaned book text into sentences using a regex-based approach.

    This is a simple heuristic splitter, not a full sentence tokenizer: it
    handles the common cases (splitting at ./!/? followed by whitespace,
    not splitting on common abbreviations or initials, not splitting mid-
    ellipsis) but will occasionally over- or under-split on unusual
    punctuation. That's an acceptable tradeoff here.
    """
    raw_pieces = _SPLIT_RE.split(text)

    sentences = []
    buffer = ""
    for piece in raw_pieces:
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        if _ends_with_abbreviation(buffer):
            continue  # hold the buffer, merge it with the next piece instead
        sentences.append(buffer)
        buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


def prepare_sequences(text: str, tokenizer: BPETokenizer, order: int) -> list[list[int]]:
    """
    Turn raw text into a list of boundary-wrapped token-ID sequences, ready
    to train or score an NGramModel of the given order: split into
    sentences, then encode each sentence with the right number of <s> /
    </s> tokens for that order.
    """
    sentences = split_into_sentences(text)
    return [tokenizer.encode_with_boundaries(sentence, order) for sentence in sentences]


class NGramModel:
    """
    An N-gram language model trained from scratch on token-ID sequences.
    Set order=1 for a unigram model, order=2 for bigram, order=3 for
    trigram -- the same class and math work for all three.
    """

    def __init__(self, order: int, vocab_size: int):
        if order < 1:
            raise ValueError("order must be at least 1 (unigram or higher)")
        self.order = order
        self.vocab_size = vocab_size

        # context_counts: how many times each (order - 1)-length context
        # tuple occurred. For a unigram model, the only context is the
        # empty tuple (), and its count is just the total token count.
        self.context_counts: dict[tuple, int] = defaultdict(int)

        # ngram_counts: how many times each full order-length tuple
        # occurred (the context plus the word that followed it).
        self.ngram_counts: dict[tuple, int] = defaultdict(int)

    def train(self, sequences: list[list[int]]) -> None:
        """
        Collect counts from a list of token-ID sequences. Each sequence
        should already include the right number of boundary tokens for
        this model's order (see prepare_sequences / encode_with_boundaries).
        """
        for sequence in sequences:
            for i in range(len(sequence) - self.order + 1):
                ngram = tuple(sequence[i:i + self.order])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

    def probability(self, context: tuple, word_id: int, k: float) -> float:
        """
        Add-k smoothed probability of word_id following the given context.
        context must be a tuple of length (order - 1); pass () for a
        unigram model.
        """
        ngram = context + (word_id,)
        ngram_count = self.ngram_counts.get(ngram, 0)
        context_count = self.context_counts.get(context, 0)
        return (ngram_count + k) / (context_count + k * self.vocab_size)

    def negative_log_prob(self, sequence: list[int], k: float) -> tuple[float, int]:
        """
        Sum of -log(probability) over every PREDICTED token in this
        sequence -- i.e. every position from index (order - 1) onward.
        The leading (order - 1) tokens are left context/padding, not
        predictions, so they're not scored on their own.
        Returns (total_negative_log_prob, number_of_predicted_tokens).
        """
        total_nlp = 0.0
        num_predicted = 0
        for i in range(len(sequence) - self.order + 1):
            ngram = tuple(sequence[i:i + self.order])
            context = ngram[:-1]
            word_id = ngram[-1]
            prob = self.probability(context, word_id, k)
            total_nlp += -math.log(prob)
            num_predicted += 1
        return total_nlp, num_predicted

    def perplexity(self, sequences: list[list[int]], k: float) -> float:
        """
        Per-word average negative log probability of a test set, per the
        assignment's definition: sum the negative log probability of every
        predicted token across ALL sequences (pooled together, not
        averaged sentence-by-sentence first), then divide by the total
        number of predicted tokens. This is NOT the exponentiated
        textbook perplexity formula.
        """
        total_nlp = 0.0
        total_predicted = 0
        for sequence in sequences:
            nlp, count = self.negative_log_prob(sequence, k)
            total_nlp += nlp
            total_predicted += count
        return total_nlp / total_predicted
