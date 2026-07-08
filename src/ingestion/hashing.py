"""Content-hash helper for incremental ingestion (see parse_pdf.py, build_index.py)."""
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
