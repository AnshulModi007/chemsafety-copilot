"""Section-aware chunking of parsed CSB reports.

Splits each report into top-level sections -- numbered "N.0 TITLE" headings
(the pattern CSB's modern investigation reports use), falling back to known
section-name keywords for older/shorter documents (case studies, safety
bulletins) that don't use that numbering -- then splits each section into
overlapping word-count chunks. Word count is used as a simple proxy for
token count (no tokenizer dependency needed at this stage).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, MANIFEST_PATH, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS  # noqa: E402

NUMBERED_HEADING_DECIMAL = re.compile(r"^(\d{1,2}\.0)\s+([A-Z][A-Za-z0-9 ,/&\-:'.]{3,90})$")
NUMBERED_HEADING_BARE = re.compile(r"^(\d{1,2})\s+([A-Z][A-Z0-9 ,/&\-:'.]{3,90})$")
KNOWN_SECTION_KEYWORDS = [
    "EXECUTIVE SUMMARY", "BACKGROUND", "INCIDENT DESCRIPTION", "KEY ISSUES",
    "ROOT CAUSE", "ROOT CAUSES", "CAUSAL FACTORS", "FACTUAL INFORMATION",
    "FINDINGS", "RECOMMENDATIONS", "REGULATORY ANALYSIS", "CONCLUSION",
    "CONCLUSIONS", "SAFETY ISSUES", "TIMELINE", "SUMMARY OF INCIDENT",
]
# Recurring page header/footer boilerplate that would otherwise false-positive
# against the bare-number heading pattern or the ALL-CAPS keyword fallback.
BOILERPLATE_SUBSTRINGS = ("CHEMICAL SAFETY AND HAZARD INVESTIGATION BOARD",)


def is_toc_line(line: str) -> bool:
    return re.search(r"\.{4,}", line) is not None


def is_boilerplate(line: str) -> bool:
    upper = line.upper()
    return any(b in upper for b in BOILERPLATE_SUBSTRINGS)


def detect_heading(line: str) -> str | None:
    line = line.strip()
    if not line or is_toc_line(line) or is_boilerplate(line):
        return None

    m = NUMBERED_HEADING_DECIMAL.match(line)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()

    m = NUMBERED_HEADING_BARE.match(line)
    if m and m.group(2) == m.group(2).upper():
        title = m.group(2)
        # Require at least one real word (not a bare regulation citation like
        # "40 CFR 68." or a stray table cell like "6 N/A 100").
        has_real_word = any(re.fullmatch(r"[A-Za-z]{4,}", tok) for tok in title.split())
        if "CFR" not in title.upper() and has_real_word:
            return f"{m.group(1)} {m.group(2)}".strip()

    if 6 < len(line) < 90 and not line.endswith((".", ",", ";")):
        norm = line.upper()
        for kw in KNOWN_SECTION_KEYWORDS:
            if norm == kw or (kw in norm and len(line.split()) <= 6):
                return line
    return None


def split_into_sections(pages: list[dict]) -> list[dict]:
    """Returns [{section, words: [(word, page_num), ...]}, ...]."""
    sections = []
    current_section = "Front Matter"
    current_words = []

    for page in pages:
        page_num = page["page"]
        for line in page["text"].split("\n"):
            heading = detect_heading(line)
            if heading:
                if current_words:
                    sections.append({"section": current_section, "words": current_words})
                current_section = heading
                current_words = []
                continue
            current_words.extend((word, page_num) for word in line.split())

    if current_words:
        sections.append({"section": current_section, "words": current_words})

    return sections


def chunk_section(section: dict, chunk_words: int, overlap_words: int) -> list[dict]:
    words = section["words"]
    chunks = []
    step = max(chunk_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start:start + chunk_words]
        if not window:
            continue
        page_nums = [p for _, p in window]
        chunks.append({
            "section": section["section"],
            "text": " ".join(w for w, _ in window),
            "page_start": min(page_nums),
            "page_end": max(page_nums),
        })
        if start + chunk_words >= len(words):
            break
    return chunks


def main():
    manifest = {e["report_id"]: e for e in json.loads(MANIFEST_PATH.read_text())}
    all_chunks = []

    for report_id, entry in manifest.items():
        parsed_path = PROCESSED_DIR / f"{report_id}.json"
        if not parsed_path.exists():
            print(f"WARNING: no parsed text for {report_id}, run parse_pdf.py first")
            continue
        parsed = json.loads(parsed_path.read_text())
        sections = split_into_sections(parsed["pages"])
        report_chunks = []
        for section in sections:
            for chunk in chunk_section(section, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS):
                if len(chunk["text"].split()) < 20:
                    continue  # skip near-empty fragments (figure/table-only sections)
                chunk_id = f"{report_id}__{len(report_chunks):04d}"
                report_chunks.append({
                    "chunk_id": chunk_id,
                    "report_id": report_id,
                    "report_title": entry["title"],
                    "chemical": entry["chemical"],
                    "incident_type": entry["incident_type"],
                    "industry": entry["industry"],
                    "year": entry["year"],
                    **chunk,
                })
        all_chunks.extend(report_chunks)
        print(f"{report_id}: {len(sections)} sections -> {len(report_chunks)} chunks")

    out_path = PROCESSED_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_chunks, indent=2))
    print(f"\nWrote {len(all_chunks)} chunks total -> {out_path}")


if __name__ == "__main__":
    main()
