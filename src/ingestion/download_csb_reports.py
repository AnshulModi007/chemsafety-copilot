"""Download a diverse set of CSB investigation report PDFs and build data/manifest.json.

Case list is curated by hand (not scraped) for diversity across incident type,
chemical, industry, and era -- scraping the index page reliably would need to
handle pagination/JS and still wouldn't guarantee a balanced sample.
"""
import json
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_PDF_DIR, MANIFEST_PATH  # noqa: E402

BASE_URL = "https://www.csb.gov"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ChemSafetyCopilot/0.1"}

CASES = [
    {"slug": "west-fertilizer-explosion-and-fire-", "chemical": "ammonium nitrate", "incident_type": "explosion", "industry": "fertilizer storage", "year": 2013},
    {"slug": "imperial-sugar-company-dust-explosion-and-fire", "chemical": "sugar dust", "incident_type": "dust explosion", "industry": "food processing", "year": 2008},
    {"slug": "chevron-richmond-refinery-fire", "chemical": "hydrocarbon (sulfur corrosion)", "incident_type": "fire", "industry": "oil refining", "year": 2012},
    {"slug": "bp-america-texas-city-refinery-explosion", "chemical": "hydrocarbon (raffinate)", "incident_type": "explosion", "industry": "oil refining", "year": 2005},
    {"slug": "bayer-cropscience-pesticide-waste-tank-explosion", "chemical": "methomyl / pesticide waste", "incident_type": "explosion", "industry": "pesticide manufacturing", "year": 2008},
    {"slug": "dupont-la-porte-facility-toxic-chemical-release-", "chemical": "methyl mercaptan", "incident_type": "toxic release", "industry": "chemical manufacturing", "year": 2006},
    {"slug": "mgpi-processing-inc-toxic-chemical-release-", "chemical": "chlorine / sulfuric acid", "incident_type": "toxic release", "industry": "chemical manufacturing", "year": 2016},
    {"slug": "freedom-industries-chemical-release-", "chemical": "crude MCHM", "incident_type": "toxic release", "industry": "chemical storage", "year": 2014},
    {"slug": "tesoro-anacortes-refinery-fatal-explosion-and-fire-", "chemical": "hydrogen / naphtha", "incident_type": "explosion", "industry": "oil refining", "year": 2010},
    {"slug": "ab-specialty-silicones-llc", "chemical": "hydrogen / silicone", "incident_type": "explosion", "industry": "specialty chemicals", "year": 2019},
    {"slug": "watson-grinding-fatal-explosion-and-fire-", "chemical": "propylene", "incident_type": "explosion", "industry": "manufacturing", "year": 2020},
    {"slug": "intercontinental-terminals-company-itc-tank-fire", "chemical": "naphtha / pyrolysis gasoline", "incident_type": "fire", "industry": "chemical storage terminal", "year": 2019},
    {"slug": "loy-lange-box-company-pressure-vessel-explosion-", "chemical": "steam (pressure vessel)", "incident_type": "explosion", "industry": "box manufacturing", "year": 2017},
    {"slug": "t2-laboratories-inc-reactive-chemical-explosion", "chemical": "MCMT intermediate (runaway reaction)", "incident_type": "explosion", "industry": "chemical manufacturing", "year": 2007},
    {"slug": "concept-sciences-hydroxylamine-explosion", "chemical": "hydroxylamine", "incident_type": "explosion", "industry": "chemical manufacturing", "year": 1999},
    {"slug": "formosa-plastics-vinyl-chloride-explosion", "chemical": "vinyl chloride", "incident_type": "explosion", "industry": "plastics manufacturing", "year": 2004},
    {"slug": "millard-refrigerated-services-ammonia-release", "chemical": "anhydrous ammonia", "incident_type": "toxic release", "industry": "cold storage / food", "year": 2010},
    {"slug": "honeywell-geismar-chlorine-and-hydrogen-fluoride-releases", "chemical": "chlorine / hydrogen fluoride", "incident_type": "toxic release", "industry": "chemical manufacturing", "year": 2020},
    {"slug": "didion-milling-company-explosion-and-fire-", "chemical": "corn dust", "incident_type": "dust explosion", "industry": "grain milling", "year": 2017},
    {"slug": "arkema-inc-chemical-plant-fire-", "chemical": "organic peroxides", "incident_type": "fire", "industry": "chemical manufacturing", "year": 2017},
]


def _get_with_retries(url: str, attempts: int = 3, backoff: float = 3.0, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=kwargs.pop("timeout", 30), **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def find_report_pdf_url(case_page_url: str) -> str | None:
    resp = _get_with_retries(case_page_url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    # CSB report PDFs are served with cache-busting query strings
    # (e.g. ".../Report_Final.pdf?13900"), so match ".pdf" anywhere in the
    # path portion of the href rather than requiring it as a suffix.
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        path = href.split("?", 1)[0]
        # Direct PDFs (".../Report.pdf?12345") and CSB's document-viewer
        # redirect links ("/file.aspx?DocumentId=NNNN", which serve the PDF
        # directly with no redirect) are both valid report locations.
        is_pdf = path.lower().endswith(".pdf")
        is_doc_viewer = "file.aspx" in href.lower() and "documentid=" in href.lower()
        if not (is_pdf or is_doc_viewer):
            continue
        if not href.startswith("http"):
            href = BASE_URL + href if href.startswith("/") else f"{BASE_URL}/{href}"
        text = a.get_text(" ", strip=True).lower()
        candidates.append((href, text))

    if not candidates:
        return None

    def score(item):
        href, text = item
        low_href = href.lower()
        s = 0
        if "final investigation report" in text:
            s += 100
        elif "final report" in text:
            s += 90
        elif "investigation report" in text:
            s += 80
        elif "report" in text:
            s += 40
        if "/assets/1/" in low_href:
            s += 20
        bad_href = ("recommendation/status_change", "testimony", "transcript", "report_schedule", "closure_plan")
        bad_text = ("testimony", "transcript", "status change", "recommendation", "release schedule", "closure plan")
        if any(bad in low_href for bad in bad_href):
            s -= 100
        if any(bad in text for bad in bad_text):
            s -= 100
        if "appendix" in text or "appendix" in low_href:
            s -= 200
        return s

    best_href, best_text = max(candidates, key=score)
    if score((best_href, best_text)) <= 0:
        return None
    return best_href


def download_pdf(url: str, dest: Path) -> bool:
    try:
        resp = _get_with_retries(url, headers=HEADERS, timeout=60)
        if resp.headers.get("Content-Type", "").lower().find("pdf") == -1 and not resp.content[:4] == b"%PDF":
            print(f"  WARNING: response for {url} doesn't look like a PDF, skipping")
            return False
        dest.write_bytes(resp.content)
        return True
    except requests.RequestException as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


def main():
    manifest = []
    for i, case in enumerate(CASES, start=1):
        report_id = f"csb_{i:02d}_{case['slug'].strip('-')}"
        case_url = f"{BASE_URL}/{case['slug']}/"
        print(f"[{i}/{len(CASES)}] {case['slug']}")

        time.sleep(1.5)  # be polite / avoid rate-limiting on csb.gov
        try:
            pdf_url = find_report_pdf_url(case_url)
        except requests.RequestException as e:
            print(f"  ERROR fetching case page: {e}")
            continue

        if not pdf_url:
            print("  WARNING: no PDF link found on case page, skipping")
            continue

        dest_path = RAW_PDF_DIR / f"{report_id}.pdf"
        if dest_path.exists():
            print(f"  already downloaded: {dest_path.name}")
        else:
            print(f"  downloading {pdf_url}")
            ok = download_pdf(pdf_url, dest_path)
            if not ok:
                continue
            time.sleep(1)  # be polite to csb.gov

        manifest.append({
            "report_id": report_id,
            "title": case["slug"].replace("-", " ").strip().title(),
            "case_page_url": case_url,
            "pdf_url": pdf_url,
            "pdf_path": str(dest_path),
            "chemical": case["chemical"],
            "incident_type": case["incident_type"],
            "industry": case["industry"],
            "year": case["year"],
        })

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)}/{len(CASES)} reports to manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
