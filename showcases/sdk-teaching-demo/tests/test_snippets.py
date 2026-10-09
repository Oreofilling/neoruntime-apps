"""Teaching payload integrity: anchors stay verbatim against the real
source, bilingual fields stay complete, links stay https.

The anchors are the point: if teaching_app.py refactors one of the calls
a station teaches, the matching assert here fails and forces the lesson
on the page to be rewritten instead of silently rotting.
"""

import json
from pathlib import Path

import pytest

from station_snippets import (
    ERROR_LESSONS,
    NEXT_STEPS,
    STATION_SNIPPETS,
    payload,
)

SHOWCASE_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sources():
    """{filename: text} for every file the snippets reference."""
    out = {}
    for st in STATION_SNIPPETS:
        if st["file"] not in out:
            out[st["file"]] = (SHOWCASE_DIR / st["file"]).read_text(
                encoding="utf-8")
    return out


@pytest.mark.parametrize("station", STATION_SNIPPETS, ids=lambda s: s["id"])
def test_anchors_exist_verbatim_in_source(station, sources):
    src = sources[station["file"]]
    for anchor in station["anchors"]:
        assert anchor in src, (
            f"{station['id']}: anchor no longer verbatim in "
            f"{station['file']}: {anchor!r} — the lesson on the page now "
            f"lies; update station_snippets.py to match the refactor")


def test_station_ids_are_exactly_s1_through_s5():
    assert [s["id"] for s in STATION_SNIPPETS] == [f"s{i}" for i in range(1, 6)]


@pytest.mark.parametrize("station", STATION_SNIPPETS, ids=lambda s: s["id"])
def test_bilingual_fields_complete(station):
    for field in ("title", "note"):
        for lang in ("en", "zh"):
            assert station[field][lang].strip(), f"{station['id']}.{field}.{lang}"


@pytest.mark.parametrize("station", STATION_SNIPPETS, ids=lambda s: s["id"])
def test_doc_links_https_known_hosts(station):
    for label, url in station["docs"]:
        assert url.startswith("https://"), url
        assert url.startswith(
            ("https://pypi.org/", "https://github.com/")), url


def test_error_lessons_attach_to_real_stations_and_are_bilingual():
    ids = {s["id"] for s in STATION_SNIPPETS}
    assert ERROR_LESSONS, "the error lessons are half the teaching value"
    for lesson in ERROR_LESSONS:
        assert lesson["station"] in ids
        for field in ("title", "cause", "contract"):
            for lang in ("en", "zh"):
                assert lesson[field][lang].strip()


def test_next_steps_shape():
    assert NEXT_STEPS["title"]["en"].strip()
    assert NEXT_STEPS["title"]["zh"].strip()
    assert 3 <= len(NEXT_STEPS["steps"]) <= 6
    for step in NEXT_STEPS["steps"]:
        assert step["en"].strip() and step["zh"].strip()
    for label, url in NEXT_STEPS["links"]:
        assert url.startswith("https://"), url


def test_payload_shape_and_json_serializable():
    data = payload()
    assert set(data) == {"stations", "errors", "next"}
    assert data["stations"] is STATION_SNIPPETS
    assert data["errors"] is ERROR_LESSONS
    assert data["next"] is NEXT_STEPS
    json.dumps(data)  # /api/snippets serves exactly this structure
