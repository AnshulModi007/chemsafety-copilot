"""Extract per-page text and tables from CSB report PDFs.

Uses pdfplumber (pure Python, wraps pdfminer.six) for both text and tables.
PyMuPDF was tried first per the original plan but its compiled extension is
blocked by this machine's Application Control policy (unsigned native DLL);
pdfplumber avoids that entirely at the cost of being somewhat slower.
"""
import json
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, MANIFEST_PATH  # noqa: E402
from src.ingestion.hashing import file_sha256  # noqa: E402

# Tracks the source PDF hash each report was last parsed from, so re-running
# this script after adding/changing a handful of PDFs only re-parses those --
# not the whole corpus -- while still catching a changed PDF that kept the
# same filename (existence-only skip logic can't tell those apart).
PARSE_HASHES_PATH = PROCESSED_DIR / "parse_hashes.json"


def parse_report(pdf_path: Path) -> dict:
    pages = []
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            pages.append({"page": page_num, "text": page.extract_text() or ""})
            for t_idx, table in enumerate(page.extract_tables() or []):
                tables.append({"page": page_num, "table_index": t_idx, "rows": table})
    return {"pages": pages, "tables": tables}


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    hashes = json.loads(PARSE_HASHES_PATH.read_text()) if PARSE_HASHES_PATH.exists() else {}

    for entry in manifest:
        report_id = entry["report_id"]
        pdf_path = Path(entry["pdf_path"])
        out_path = PROCESSED_DIR / f"{report_id}.json"
        current_hash = file_sha256(pdf_path)

        if out_path.exists() and hashes.get(report_id) == current_hash:
            print(f"skip (unchanged): {report_id}")
            continue

        print(f"parsing {report_id} ({pdf_path.name})")
        parsed = parse_report(pdf_path)
        parsed["report_id"] = report_id
        out_path.write_text(json.dumps(parsed))
        hashes[report_id] = current_hash
        print(f"  {len(parsed['pages'])} pages, {len(parsed['tables'])} tables")

    PARSE_HASHES_PATH.write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
