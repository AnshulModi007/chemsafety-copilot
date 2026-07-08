"""Live chemical property lookups against PubChem's free, no-key PUG REST /
PUG View APIs -- used for questions about a specific chemical's properties
rather than historical-incident questions (which go through RAG instead).
"""
import logging
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PUBCHEM_BASE_URL, PUBCHEM_VIEW_URL  # noqa: E402

logger = logging.getLogger(__name__)

PROPERTIES = [
    "MolecularFormula", "MolecularWeight", "IUPACName", "CanonicalSMILES",
    "XLogP", "TPSA", "HBondDonorCount", "HBondAcceptorCount",
]


class CompoundNotFound(Exception):
    """PubChem responded, but has no compound matching the given name."""


class PubChemUnavailable(Exception):
    """PubChem could not be reached, timed out, or returned an unexpected
    (malformed / restructured) response -- distinct from CompoundNotFound so
    callers can tell "no such chemical" from "the lookup itself failed"."""


def _get_cid(name: str) -> int:
    """Resolve a compound name to its PubChem CID.

    Raises:
        CompoundNotFound: PubChem has no match for this name.
        PubChemUnavailable: the request timed out, PubChem is unreachable,
            or the response wasn't in the expected shape.
    """
    url = f"{PUBCHEM_BASE_URL}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    try:
        resp = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        raise PubChemUnavailable(f"Could not reach PubChem: {e}") from e
    if resp.status_code == 404:
        raise CompoundNotFound(f"PubChem has no compound matching '{name}'")
    try:
        resp.raise_for_status()
        return resp.json()["IdentifierList"]["CID"][0]
    except requests.exceptions.HTTPError as e:
        raise PubChemUnavailable(f"PubChem returned an error: {e}") from e
    except (ValueError, KeyError, IndexError) as e:
        raise PubChemUnavailable(f"PubChem returned an unexpected response shape: {e}") from e


def _get_properties(cid: int) -> dict:
    """Fetch the fixed PROPERTIES list for a CID.

    Raises:
        PubChemUnavailable: the request timed out, PubChem is unreachable,
            or the response wasn't in the expected shape.
    """
    prop_list = ",".join(PROPERTIES)
    url = f"{PUBCHEM_BASE_URL}/compound/cid/{cid}/property/{prop_list}/JSON"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()["PropertyTable"]["Properties"][0]
    except requests.exceptions.RequestException as e:
        raise PubChemUnavailable(f"Could not reach PubChem: {e}") from e
    except (ValueError, KeyError, IndexError) as e:
        raise PubChemUnavailable(f"PubChem returned an unexpected response shape: {e}") from e


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
    here means "not available from PubChem", not "no hazards". This is
    supplementary to the core property lookup, so network/timeout/malformed-
    response failures fail soft to an empty list (logged) rather than raising --
    a missing hazard row shouldn't block the rest of the answer.
    """
    try:
        url = f"{PUBCHEM_VIEW_URL}/data/compound/{cid}/JSON?heading=GHS+Classification"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        logger.warning("GHS hazard lookup failed for CID %s", cid, exc_info=True)
        return []

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
