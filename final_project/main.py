
"""
Final project implementation.
"""

# pylint: disable=wrong-import-position
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lab_6_pipeline.pipeline import UDPipeAnalyzer


def main(corpus_path: Path, dist_path: Path) -> None:
    """
    Generate conllu file for provided corpus of texts.
 
    Args:
        corpus_path (Path): Path to folder containing text files.
        dist_path (Path): Path to folder for saving auto_annotated.conllu.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus folder does not exist: {corpus_path}")

    raw_files = sorted(corpus_path.glob("*.txt"), key=sort_key)

    if not raw_files:
        raise ValueError(f"No .txt files found in {corpus_path}")

    dist_path.mkdir(parents=True, exist_ok=True)

    texts = [f.read_text(encoding="utf-8").strip() for f in raw_files]
    texts = [t for t in texts if t]

    (dist_path / "corpus.txt").write_text("\n\n".join(texts), encoding="utf-8")

    analyzer = UDPipeAnalyzer()

    conllu_blocks: list[str] = []
    for text in texts:
        annotated = analyzer.analyze([text])
        if annotated:
            conllu_blocks.append(annotated[0])

    counter = 1
    renumbered = []
    for line in "".join(conllu_blocks).split("\n"):
        if line.startswith("# sent_id"):
            renumbered.append(f"# sent_id = {counter}")
            counter += 1
        else:
            renumbered.append(line)

    result = "\n".join(renumbered)
    assert result, "Result is None"

    (dist_path / "auto_annotated.conllu").write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main(Path(__file__).parent / "assets" / "articles", Path(__file__).parent / "dist")
