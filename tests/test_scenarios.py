"""Tests for scenarios.py — CRUD round-trip and NaN sanitization."""

import json
import math

import pytest

import scenarios as SC


@pytest.fixture
def scen_file(tmp_path, monkeypatch):
    path = tmp_path / "scenarios.json"
    monkeypatch.setattr(SC, "SCENARIOS_FILE", str(path))
    return path


VALID = {
    "tickers": ["VOO", "GLD"],
    "weights": [0.7, 0.3],
    "expenses": [0.0003, 0.004],
}


def test_sanitize_strips_nan_rows():
    dirty = {
        "tickers": ["VOO", "", "GLD"],
        "weights": [0.5, float("nan"), 0.5],
        "expenses": [0.001, 0.002, 0.003],
    }
    clean = SC._sanitize_scenario(dirty)
    assert clean["tickers"] == ["VOO", "GLD"]
    assert clean["weights"] == [0.5, 0.5]
    assert all(math.isfinite(w) for w in clean["weights"])
    assert all(isinstance(e, (int, float)) and math.isfinite(e) for e in clean["expenses"])


def test_sanitize_keeps_valid_row_alignment():
    """A NaN in one row removes the whole row — weights stay paired."""
    dirty = {
        "tickers": ["AAA", "", "CCC"],
        "weights": [0.2, 0.8, 0.0],
        "expenses": [0.1, 0.9, 0.01],
    }
    clean = SC._sanitize_scenario(dirty)
    assert clean == {"tickers": ["AAA", "CCC"], "weights": [0.2, 0.0],
                     "expenses": [0.1, 0.01]}


def test_create_and_roundtrip(scen_file):
    store = {}
    err = SC.create_scenario(store, "My Port", **VALID)
    assert err is None
    assert scen_file.exists()
    loaded = SC.load_scenarios()
    assert loaded["My Port"]["tickers"] == VALID["tickers"]


def test_create_rejects_bad_weight_sum(scen_file):
    store = {}
    bad = {**VALID, "weights": [0.5, 0.4]}
    assert SC.create_scenario(store, "Bad", **bad) is not None
    assert "Bad" not in store


def test_create_rejects_duplicate_name(scen_file):
    store = {}
    assert SC.create_scenario(store, "X", **VALID) is None
    assert SC.create_scenario(store, "X", **VALID) is not None


def test_update_and_delete(scen_file):
    store = {}
    SC.create_scenario(store, "P", **VALID)
    new = {"tickers": ["QQQ"], "weights": [1.0], "expenses": [0.0]}
    assert SC.update_scenario(store, "P", **new) is None
    assert store["P"]["tickers"] == ["QQQ"]

    SC.delete_scenario(store, "P")
    assert "P" not in store
    assert "P" not in json.loads(scen_file.read_text())


def test_update_missing_returns_error(scen_file):
    store = {}
    assert SC.update_scenario(store, "ghost", **VALID) is not None


def test_import_export_roundtrip(scen_file, tmp_path):
    # Import into a fresh store backed by a separate file to avoid name clash
    monkey_target = tmp_path / "other.json"
    import scenarios
    original = scenarios.SCENARIOS_FILE
    scenarios.SCENARIOS_FILE = str(monkey_target)
    try:
        store = {}
        exported = json.dumps({"Imported": VALID})
        imported, skipped = SC.import_scenarios(store, exported)
        assert (imported, skipped) == (1, 0)
        assert store["Imported"]["tickers"] == VALID["tickers"]
        # Re-import same → skipped
        imported2, skipped2 = SC.import_scenarios(store, exported)
        assert (imported2, skipped2) == (0, 1)
        blob = SC.export_scenarios(store, ["Imported"])
        assert json.loads(blob)["Imported"]["weights"] == VALID["weights"]
    finally:
        scenarios.SCENARIOS_FILE = original


def test_load_seeds_defaults_when_missing(scen_file):
    loaded = SC.load_scenarios()
    assert isinstance(loaded, dict) and len(loaded) > 0
    for details in loaded.values():
        assert len(details["tickers"]) == len(details["weights"]) == len(details["expenses"])
