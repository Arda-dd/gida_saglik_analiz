"""Bilgi tabani (data/knowledge_base/docs/*.md) dokumanlarini RAG icin parcalara (chunk) ayirir.

Her dokuman bir H1 basligi + dogrulama metadata bloğu (**Durum:**/**Kaynak:**/**URL:**) ile
baslar, ardindan "## " ile ayrilan bolumler gelir. Her bolum ayri bir chunk olur; metadata
(verified, source, source_url) tum chunk'lara dokuman seviyesinde miras kalir - boylece retriever
her sonuc icin kaynagin dogrulanmis olup olmadigini (guven skoru) bilir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DOCS_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base" / "docs"

_META_LINE = re.compile(r"^\*\*(?P<key>[^:*]+):\*\*\s*(?P<value>.+)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    verified: bool
    source: str | None
    source_url: str | None

    @property
    def embedding_text(self) -> str:
        """Embedding modeline verilecek, baglam icin baslik/bolum onekli metin."""
        return f"{self.title} — {self.section}\n{self.text}"


def _parse_doc_metadata(meta_block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in meta_block.splitlines():
        match = _META_LINE.match(line.strip())
        if match:
            fields[match.group("key").strip().lower()] = match.group("value").strip()
    return fields


def _is_verified(durum_value: str | None) -> bool:
    if not durum_value:
        return False
    return durum_value.strip().upper().startswith("DOĞRULANDI")


def chunk_markdown_file(path: Path) -> list[Chunk]:
    """Tek bir markdown dokumanini bolum (## ...) bazinda chunk'lara ayirir."""
    raw = path.read_text(encoding="utf-8")
    doc_id = path.stem

    h1_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = h1_match.group(1).strip() if h1_match else doc_id

    body_start = h1_match.end() if h1_match else 0
    remainder = raw[body_start:]

    first_h2 = re.search(r"^##\s+", remainder, re.MULTILINE)
    meta_block = remainder[: first_h2.start()] if first_h2 else remainder
    meta_fields = _parse_doc_metadata(meta_block)

    verified = _is_verified(meta_fields.get("durum"))
    source = meta_fields.get("kaynak")
    source_url = meta_fields.get("url")

    # Metadata satirlari (**Durum:** vb.) disindaki govde (ornegin glossary dokumanindaki
    # TR/EN terim tablosu) bilgi tasiyabilir - sadece meta alanlari cikarilip geri kalani
    # ayri bir "preamble" chunk'i olarak saklanir, boylece hicbir icerik sessizce kaybolmaz.
    preamble_lines = [
        line for line in meta_block.splitlines() if not _META_LINE.match(line.strip())
    ]
    preamble_body = "\n".join(preamble_lines).strip()

    sections_text = remainder[first_h2.start():] if first_h2 else ""
    section_matches = list(re.finditer(r"^##\s+(.+)$", sections_text, re.MULTILINE))

    chunks: list[Chunk] = []
    if preamble_body:
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::preamble",
                doc_id=doc_id,
                title=title,
                section=title,
                text=preamble_body,
                verified=verified,
                source=source,
                source_url=source_url,
            )
        )

    for i, sec_match in enumerate(section_matches):
        section_title = sec_match.group(1).strip()
        content_start = sec_match.end()
        content_end = (
            section_matches[i + 1].start() if i + 1 < len(section_matches) else len(sections_text)
        )
        section_body = sections_text[content_start:content_end].strip()
        if not section_body:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::{i}",
                doc_id=doc_id,
                title=title,
                section=section_title,
                text=section_body,
                verified=verified,
                source=source,
                source_url=source_url,
            )
        )
    return chunks


def chunk_knowledge_base(docs_dir: Path = KB_DOCS_DIR) -> list[Chunk]:
    """Bilgi tabanindaki tum .md dokumanlarini chunk'layip birlestirir."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        chunks.extend(chunk_markdown_file(path))
    return chunks
