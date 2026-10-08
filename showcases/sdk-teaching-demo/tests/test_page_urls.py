"""Static guards on the page: every JS-built URL goes through appUrl(),
no external resources, bilingual dictionary present, five stations."""

from pathlib import Path

INDEX = Path(__file__).resolve().parent.parent / "templates" / "index.html"


def read():
    return INDEX.read_text(encoding="utf-8")


def test_appurl_defined_and_used_for_every_fetch():
    page = read()
    assert "function appUrl(path)" in page
    for line in page.splitlines():
        stripped = line.strip()
        if "fetch(" in stripped:
            assert "appUrl(" in stripped, f"bare fetch: {stripped}"


def test_mjpeg_src_goes_through_appurl():
    page = read()
    assert 'appUrl("/stream.mjpg")' in page


def test_no_external_resources():
    page = read()
    assert "https://" not in page
    assert "http://" not in page
    assert "<link" not in page  # no stylesheets beyond the inline one


def test_bilingual_dictionary_covers_stations():
    page = read()
    assert "var I18N" in page
    assert "en: {}" in page and "zh: {" in page
    for key in ("s1Title", "s2Title", "s3Title", "s4Title", "s5Title",
                "feedOff", "refusalBtn"):
        assert key in page, f"missing i18n key {key}"
    assert page.count("data-i18n") >= 12
    assert "teach-lang" in page  # localStorage persistence


def test_five_station_sections_present():
    page = read()
    for station_id in ("s1", "s2", "s3", "s4", "s5"):
        assert f'id="{station_id}"' in page
    # the console-proxy survival rule is stated where it's implemented
    assert "reverse proxy" in page
