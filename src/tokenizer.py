"""
BPE tokenizer wrapper for this project.

This file is the ONLY place in the project that imports the Hugging Face
`tokenizers` library. Everything else imports the BPETokenizer class from
here, which is what the assignment means by "provide an interface to the
Hugging Face code."

Uses HF's own BPE *implementation* (not a pre-trained tokenizer) — we build
an empty tokenizer and train it ourselves on our own cleaned text files.
"""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers

# Special tokens every trained tokenizer will have.
# [UNK]  -> stand-in for any token the vocabulary doesn't cover.
# <s>    -> sentence/sequence start marker, used later for N-gram context.
# </s>   -> sentence/sequence end marker.
UNK_TOKEN = "[UNK]"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
SPECIAL_TOKENS = [UNK_TOKEN, START_TOKEN, END_TOKEN]

# Maps a friendly string name (what callers pass in) to the actual HF
# pre-tokenizer object. Keeping this mapping here means callers never need
# to import anything from `tokenizers` themselves.
PRE_TOKENIZER_CHOICES = {
    "whitespace": pre_tokenizers.Whitespace,
    "whitespace_split": pre_tokenizers.WhitespaceSplit,
    "byte_level": pre_tokenizers.ByteLevel,
}


class BPETokenizer:
    """
    A trainable byte-pair-encoding tokenizer, built on Hugging Face's BPE
    implementation, with the specific interface this project needs:
    train on our own files, encode/decode, and save/load a trained model.
    """

    def __init__(self, vocab_size: int, pre_tokenizer: str = "whitespace", min_frequency: int = 2):
        """
        vocab_size:     target vocabulary size (actual trained size may be
                         slightly smaller — see the `vocab_size` property).
        pre_tokenizer:  one of PRE_TOKENIZER_CHOICES, e.g. "whitespace".
        min_frequency:  minimum number of occurrences a pair must have
                         before BPE will consider merging it.
        """
        if pre_tokenizer not in PRE_TOKENIZER_CHOICES:
            valid = ", ".join(PRE_TOKENIZER_CHOICES)
            raise ValueError(f"pre_tokenizer must be one of: {valid}")

        self._requested_vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._pre_tokenizer_name = pre_tokenizer

        # Build an untrained tokenizer: a BPE model (with our unknown-token
        # placeholder) plus the chosen pre-tokenizer. Nothing is trained yet.
        self._tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        self._tokenizer.pre_tokenizer = PRE_TOKENIZER_CHOICES[pre_tokenizer]()

        # Cached special-token IDs, filled in once training has happened
        # (token_to_id only works after training builds the vocabulary).
        self._unk_id = None
        self._start_id = None
        self._end_id = None

    def train(self, file_paths: list[str]) -> None:
        """
        Train the BPE vocabulary on the given text files (e.g. our cleaned
        data/clean/hobbit.txt and data/clean/lostworld.txt). Training on
        both files together, in one call, produces ONE shared vocabulary
        rather than one per file.
        """
        trainer = BpeTrainer(
            vocab_size=self._requested_vocab_size,
            min_frequency=self._min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        file_paths = [str(p) for p in file_paths]
        self._tokenizer.train(file_paths, trainer)

        # Now that training has happened, look up and cache the special
        # token IDs so encode_with_boundaries() doesn't repeat this lookup.
        self._unk_id = self._tokenizer.token_to_id(UNK_TOKEN)
        self._start_id = self._tokenizer.token_to_id(START_TOKEN)
        self._end_id = self._tokenizer.token_to_id(END_TOKEN)

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of vocabulary IDs."""
        return self._tokenizer.encode(text).ids

    def encode_tokens(self, text: str) -> list[str]:
        """
        Convert text into a list of readable token strings (not IDs).
        Only meant for debugging/inspection, not used by the N-gram models.
        """
        return self._tokenizer.encode(text).tokens

    def decode(self, ids: list[int]) -> str:
        """Convert a list of vocabulary IDs back into text."""
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """The ACTUAL trained vocabulary size (may be <= the requested one)."""
        return self._tokenizer.get_vocab_size()

    @property
    def unk_id(self) -> int:
        """The vocabulary ID of the [UNK] token, useful for measuring UNK rate."""
        return self._unk_id

    def save(self, path: str) -> None:
        """Save the trained tokenizer to a JSON file so it can be reloaded."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a previously-trained tokenizer from a JSON file."""
        # We don't know the original constructor arguments from the file
        # alone, so build a bare instance and swap in the loaded tokenizer.
        instance = cls.__new__(cls)
        instance._tokenizer = Tokenizer.from_file(str(path))
        instance._requested_vocab_size = instance._tokenizer.get_vocab_size()
        instance._min_frequency = None
        instance._pre_tokenizer_name = None
        instance._unk_id = instance._tokenizer.token_to_id(UNK_TOKEN)
        instance._start_id = instance._tokenizer.token_to_id(START_TOKEN)
        instance._end_id = instance._tokenizer.token_to_id(END_TOKEN)
        return instance

    def encode_with_boundaries(self, text: str, n: int) -> list[int]:
        """
        Encode text and wrap it with sentence-boundary token IDs, for use
        as N-gram model input.

        An order-n model (unigram: n=1, bigram: n=2, trigram: n=3) needs
        (n-1) <s> tokens of left context before the first real token, and
        one </s> token marking the end, e.g. for a trigram model (n=3):
            <s> <s> w1 w2 w3 ... wk </s>
        For a unigram model (n=1), there's no context needed, so no
        leading <s> tokens are added -- just the tokens plus </s>.
        """
        if n < 1:
            raise ValueError("n must be at least 1 (unigram or higher)")

        leading_starts = [self._start_id] * (n - 1)
        body = self.encode(text)
        return leading_starts + body + [self._end_id]
