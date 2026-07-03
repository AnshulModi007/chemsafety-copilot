"""Live chemical property lookups against PubChem's free, no-key PUG REST /
PUG View APIs -- used for questions about a specific chemical's properties
rather than historical-incident questions (which go through RAG instead).
"""
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PUBCHEM_BASE_URL, PUBCHEM_VIEW_URL  # noqa: E402

PROPERTIES = [
    "MolecularFormula", "MolecularWeight", "IUPACName", "CanonicalSMILES",
    "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount",
]


class CompoundNotFound(Exception):
    pass


def _get_cid(name: str) -> int:
    url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        raise CompoundNotFound(f"PubChem has no compound matching '{name}'")
    resp.raise_for_status()
    return resp.json()["IdentifierList"]["CID"][0]


def _get_properties(cid: int) -> dict:
    prop_list = ",".join(PROPERTIES)
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/{prop_list}/JSON"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()["PropertyTable"]["Properties"][0]


_H_CODE_RE = re.compile(r"^(H\d{3}[fi]?)\s*(?:\(([^)]+)\))?:\s*(.*)$")


def _get_ghs_hazards(cid: int) -> list[str]:
    """Best-effort GHS hazard statement codes (e.g. "H272: ...") from PUG View,
    deduplicated by code.

    PubChem's GHS section mixes actual hazard statements in with per-notification
    aggregation boilerplate ("Reported as ... by N of M companies", signal-word-only
    lines, pictogram code lists) and repeats the same code across multiple
    notification cohorts, each with its own confidence percentage. We keep one
    entry per H-code (the highest-confidence variant, or the plain one if no
    cohort reports a percentage) rather than surfacing every repetition.
    Not every compound has a GHS Classification section at all, so an empty list
    here means "not available from PubChem", not "no hazards".
    """
    url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=GHS+Classification"
    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    best_by_code: dict[str, tuple[float, str]] = {}
    try:
        sections = resp.json()["Record"]["Section"]
        for section in sections:
            for sub in section.get("Section", []):
                for inner in sub.get("Section", []):
                    if inner.get("TOCHeading") != "GHS Classification":
                        continue
                    for info in inner.get("Information", []):
                        for markup in info.get("Value", {}).get("StringWithMarkup", []):
                            text = markup.get("String", "").strip()
                            m = _H_CODE_RE.match(text)
                            if not m:
                                continue
                            code, pct_str, rest = m.groups()
                            try:
                                pct = float(pct_str.rstrip("%").lstrip("> ")) if pct_str else -1.0
                            except ValueError:
                                pct = -1.0
                            if code not in best_by_code or pct > best_by_code[code][0]:
                                best_by_code[code] = (pct, f"{code}: {rest}")
    except (KeyError, IndexError):
        pass
    return [entry for _, entry in best_by_code.values()]


def get_compound_properties(name: str) -> dict:
    """Look up a chemical by common/IUPAC name. Raises CompoundNotFound if
    PubChem doesn't recognize the name -- callers should surface that to the
    user rather than let the LLM guess at properties.
    """
    cid = _get_cid(name)
    props = _get_properties(cid)
    hazards = _get_ghs_hazards(cid)

    return {
        "query": name,
        "cid": cid,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        "iupac_name": props.get("IUPACName"),
        "molecular_formula": props.get("MolecularFormula"),
        "molecular_weight": props.get("MolecularWeight"),
        "canonical_smiles": props.get("CanonicalSMILES"),
        "xlogp": props.get("XLogP"),
        "tpsa": props.get("TPSA"),
        "h_bond_donor_count": props.get("HBondDonorCount"),
        "h_bond_acceptor_count": props.get("HBondAcceptorCount"),
        "ghs_hazard_statements": hazards,
    }


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "ammonium nitrate"
    import json
    try:
        print(json.dumps(get_compound_properties(query), indent=2))
    except CompoundNotFound as e:
        print(f"Not found: {e}")
