"""Tests for the generation context budget.

Groq's free tier caps a single request at 8,000 tokens per model. Parent-child
retrieval hands generation five wide parent windows (~55k characters, ~13.7k
tokens), which exceeded that on every grounded answer. These cover the three
degradation steps and, most importantly, that no source is ever silently
dropped -- a missing chunk means a missing citation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.generate import _build_context, _select_context_bodies  # noqa: E402


def _hit(i: int, text_len: int, parent_len: int) -> dict:
    return {
        "chunk_id": f"c{i}",
        "report_id": f"report_{i}",
        "report_title": f"Report {i}",
        "section": f"Section {i}",
        "page_start": 1,
        "page_end": 2,
        "parent_page_start": 1,
        "parent_page_end": 4,
        "text": f"narrow{i}" + "n" * max(0, text_len - 7),
        "parent_text": f"wide{i}" + "w" * max(0, parent_len - 5),
    }


def test_full_parent_windows_are_used_when_they_fit():
    # The wide window is the whole point of parent-child retrieval; it must not
    # be given up while there is room for it.
    hits = [_hit(i, 100, 400) for i in range(5)]
    bodies = _select_context_bodies(hits, char_budget=20_000)
    assert bodies == [h["parent_text"] for h in hits]


def test_falls_back_to_the_retrieval_windows_when_parents_overflow():
    # 5 x 15k parent = 75k, over budget; 5 x 3k narrow = 15k, under.
    hits = [_hit(i, 3_000, 15_000) for i in range(5)]
    bodies = _select_context_bodies(hits, char_budget=20_000)
    assert bodies == [h["text"] for h in hits]
    assert sum(len(b) for b in bodies) <= 20_000


def test_truncates_proportionally_when_even_the_narrow_windows_overflow():
    hits = [_hit(i, 10_000, 20_000) for i in range(5)]
    bodies = _select_context_bodies(hits, char_budget=20_000)

    assert len(bodies) == 5
    # Budget is respected once the truncation marker is discounted.
    marker = "\n[... excerpt truncated ...]"
    assert sum(len(b) - len(marker) for b in bodies) <= 20_000
    assert all(b.endswith(marker) for b in bodies)


def test_truncation_is_marked_so_a_severed_sentence_is_not_read_as_the_end():
    hits = [_hit(0, 50_000, 50_000)]
    body = _select_context_bodies(hits, char_budget=1_000)[0]
    assert "[... excerpt truncated ...]" in body


def test_no_chunk_is_ever_dropped_however_tight_the_budget():
    # Dropping a chunk removes a citable source silently; shortening all of
    # them is the lesser harm.
    hits = [_hit(i, 10_000, 20_000) for i in range(5)]
    for budget in (20_000, 5_000, 500, 10, 1):
        assert len(_select_context_bodies(hits, char_budget=budget)) == 5


def test_the_real_world_case_now_fits_a_single_free_tier_request():
    """Measured from the live corpus: five chunks whose parent windows total
    ~54.6k chars (~13.7k tokens) against an 8,000-token cap."""
    sizes = [(4_964, 7_980), (5_112, 15_333), (1_190, 1_190), (5_067, 14_612), (5_000, 14_777)]
    hits = [_hit(i, t, p) for i, (t, p) in enumerate(sizes)]

    unbudgeted = sum(p for _, p in sizes)
    assert unbudgeted // 4 > 8_000, "premise: the raw context exceeds the cap"

    context = _build_context(hits, char_budget=20_000)
    # ~4 chars/token, plus room for the system prompt and the completion.
    assert len(context) // 4 < 6_000


def test_context_keeps_the_metadata_header_for_every_chunk():
    # Citations are only possible if report_id and page survive budgeting.
    hits = [_hit(i, 10_000, 20_000) for i in range(3)]
    context = _build_context(hits, char_budget=1_000)
    for i in range(3):
        assert f"report_id=report_{i}" in context
        assert f'title="Report {i}"' in context


def test_empty_hits_produce_empty_context():
    assert _select_context_bodies([], char_budget=20_000) == []
    assert _build_context([], char_budget=20_000) == ""


def test_a_chunk_without_a_parent_window_falls_back_to_its_text():
    # Chunks indexed before parent_text existed carry only `text`.
    legacy = {
        "chunk_id": "c", "report_id": "r", "report_title": "R", "section": "S",
        "page_start": 1, "page_end": 1, "text": "legacy chunk body",
    }
    assert _select_context_bodies([legacy], char_budget=20_000) == ["legacy chunk body"]
    assert "legacy chunk body" in _build_context([legacy], char_budget=20_000)
