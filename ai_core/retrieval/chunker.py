"""Split markdown files into chunks suitable for retrieval."""
import re
from pathlib import Path


def chunk_file(path: Path) -> list[dict]:
    """Split a markdown file by level-2 headers (## ...).

    Returns a list of {"source": filename, "section": header, "text": content}.
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    sections = re.split(r"\n(?=## )", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < 40:
            continue
        lines = section.splitlines()
        header = lines[0].lstrip("#").strip() if lines else path.stem
        chunks.append({
            "source": path.name,
            "section": header,
            "text": section[:3000],
        })
    return chunks


def chunk_knowledge_base(root: Path) -> list[dict]:
    """Recursively chunk all .md files under root."""
    chunks = []
    for md_file in sorted(root.rglob("*.md")):
        rel_source = str(md_file.relative_to(root))
        file_chunks = chunk_file(md_file)
        for chunk in file_chunks:
            chunk["source"] = rel_source
        chunks.extend(file_chunks)
    return chunks
