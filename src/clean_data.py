"""
Clean the raw training texts (data/raw/*.txt) before BPE tokenization.

Two cleaning steps:
1. Strip Project Gutenberg license/header/footer text, if present.
2. Join hard-wrapped lines back into single lines per paragraph, so a
   sentence that was split across several lines in the raw .txt file
   becomes one line, while blank lines still mark paragraph breaks.

Never modifies data/raw/*.txt — writes results to data/clean/*.txt instead.
"""

import re
from pathlib import Path

# Paths are relative to this file's location, so the script works no matter
# where it's run from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"

GUTENBERG_START_MARKER = "*** START OF THE PROJECT GUTENBERG"
GUTENBERG_END_MARKER = "*** END OF THE PROJECT GUTENBERG"


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg's license header/footer, keeping only the
    actual book text between the START and END banner lines.

    If no START/END markers are found (e.g. a file with no Gutenberg
    boilerplate at all), the text is returned unchanged.
    """
    lines = text.split("\n")

    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if GUTENBERG_START_MARKER in line:
            start_index = i
        elif GUTENBERG_END_MARKER in line:
            end_index = i
            break  # no need to keep scanning once we've found the end

    # No banners found at all -> nothing to strip, return as-is.
    if start_index is None or end_index is None:
        return text

    # Keep only the lines strictly between the two banner lines, so the
    # banner lines themselves (Gutenberg's text, not the author's) are
    # also dropped.
    book_lines = lines[start_index + 1 : end_index]
    return "\n".join(book_lines)


def join_wrapped_lines(text: str) -> str:
    """
    Rejoin hard-wrapped lines into single lines per paragraph.

    A "paragraph" here is a block of text separated from its neighbors by
    one or more blank lines. Within a paragraph, every line break is just
    where the original file happened to wrap the text, so we replace it
    with a single space. Blank lines between paragraphs are kept, so
    paragraph structure is still visible in the output.
    """
    # Split the text into paragraphs on any run of blank line(s).
    # A blank line is a line with nothing but whitespace on it.
    paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Split the paragraph into its wrapped lines, strip stray leading/
        # trailing whitespace from each (e.g. centered title lines have
        # leading spaces we don't want stuck in the middle of a sentence),
        # drop any lines that are now empty, and join what's left with a
        # single space.
        lines = [line.strip() for line in paragraph.split("\n")]
        lines = [line for line in lines if line]
        joined = " ".join(lines)
        cleaned_paragraphs.append(joined)

    # Drop paragraphs that ended up empty (e.g. from leading/trailing
    # blank lines in the original file), then put a blank line back
    # between the paragraphs we kept.
    cleaned_paragraphs = [p for p in cleaned_paragraphs if p]
    return "\n\n".join(cleaned_paragraphs)


def clean_file(input_path: Path, output_path: Path) -> tuple[str, str]:
    """
    Read a raw text file, clean it, and write the result to output_path.
    Returns (raw_text, cleaned_text) so the caller can report stats.
    """
    raw_text = input_path.read_text(encoding="utf-8")

    cleaned_text = strip_gutenberg_boilerplate(raw_text)
    cleaned_text = join_wrapped_lines(cleaned_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_text, encoding="utf-8")

    return raw_text, cleaned_text


if __name__ == "__main__":
    files_to_clean = ["hobbit.txt", "lostworld.txt"]

    for filename in files_to_clean:
        raw_text, cleaned_text = clean_file(
            RAW_DIR / filename, CLEAN_DIR / filename
        )

        print(f"=== {filename} ===")
        print(f"characters before: {len(raw_text)}")
        print(f"characters after:  {len(cleaned_text)}")
        print("first 300 characters of cleaned file:")
        print(repr(cleaned_text[:300]))
        print()
