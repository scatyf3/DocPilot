"""Turn uploaded bytes (PDF / HTML / Markdown) into Markdown-ish text plus a title.

Everything is normalised to Markdown so a single heading-aware chunker handles all formats.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}


@dataclass
class ParsedDoc:
    title: str
    text: str


def parse_document(filename: str, data: bytes) -> ParsedDoc:
    suffix = PurePath(filename).suffix.lower()
    stem = PurePath(filename).stem
    if suffix not in SUPPORTED:
        raise ValueError(f"unsupported file type {suffix!r}; expected one of {sorted(SUPPORTED)}")
    if suffix == ".pdf":
        return _parse_pdf(data, stem)
    text = data.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        return _parse_html(text, stem)
    return ParsedDoc(title=_first_heading(text) or stem, text=text)


def _first_heading(markdown: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _parse_pdf(data: bytes, fallback_title: str) -> ParsedDoc:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    title = (reader.metadata.title if reader.metadata else None) or fallback_title
    return ParsedDoc(title=title, text="\n\n".join(p for p in pages if p))


def _parse_html(html: str, fallback_title: str) -> ParsedDoc:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    # Keep headings as Markdown '#' lines so the chunker can split on them.
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text or el.find_parent(["p", "li", "pre", "td"]):
            continue  # empty, or already captured by its block-level parent
        if el.name.startswith("h"):
            lines.append(f"{'#' * int(el.name[1])} {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else fallback_title
    return ParsedDoc(title=title or fallback_title, text="\n\n".join(lines))
