"""Central settings for ChemSafety Copilot, loaded from .env (see .env.example)."""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_POOL = 20  # candidates pulled from hybrid search before reranking down to TOP_K

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_PDF_DIR = Path(os.getenv("RAW_PDF_DIR", DATA_DIR / "raw_pdfs"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed"))
MANIFEST_PATH = DATA_DIR / "manifest.json"

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "csb_reports")

EVAL_DIR = PROJECT_ROOT / "src" / "eval"
GOLDEN_QA_PATH = EVAL_DIR / "golden_qa.json"
BASELINE_METRICS_PATH = PROJECT_ROOT / "baseline_metrics.json"

CHUNK_TOKENS = 768
CHUNK_OVERLAP_TOKENS = 100
TOP_K = 5

PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"

for _dir in (RAW_PDF_DIR, PROCESSED_DIR, CHROMA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
