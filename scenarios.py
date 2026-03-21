"""
Scenario persistence — load, save, create, delete.
"""

import json
import os

import streamlit as st

from config import DEFAULT_EXPENSE_RATIOS, INITIAL_SCENARIOS

SCENARIOS_FILE = "scenarios.json"


def load_scenarios() -> dict:
    """Load scenarios from JSON file or initialize with defaults."""
    if os.path.exists(SCENARIOS_FILE):
        try:
            with open(SCENARIOS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading scenarios: {e}")
            return {}

    # First run — seed from built-in defaults
    scenarios = {}
    for name, (tickers, weights) in INITIAL_SCENARIOS.items():
        expenses = [DEFAULT_EXPENSE_RATIOS.get(t, 0.0) for t in tickers]
        scenarios[name] = {
            "tickers": tickers,
            "weights": weights,
            "expenses": expenses,
        }
    save_scenarios(scenarios)
    return scenarios


def save_scenarios(scenarios: dict) -> None:
    """Persist scenarios to JSON file."""
    try:
        with open(SCENARIOS_FILE, 'w') as f:
            json.dump(scenarios, f, indent=4)
    except Exception as e:
        st.error(f"Error saving scenarios: {e}")


def ensure_expenses(details: dict) -> dict:
    """Back-fill expense ratios for older scenario entries."""
    if "expenses" not in details:
        details["expenses"] = [
            DEFAULT_EXPENSE_RATIOS.get(t, 0.0) for t in details["tickers"]
        ]
    return details


def create_scenario(
    scenarios: dict,
    name: str,
    tickers: list[str],
    weights: list[float],
    expenses: list[float],
) -> str | None:
    """Validate and add a new scenario. Returns error message or None on success."""
    if not tickers:
        return "Please add at least one ticker."
    total = sum(weights)
    if abs(total - 1.0) > 0.01:
        return f"Weights must sum to 1.0 (current: {total:.2f})."
    if name in scenarios:
        return "A scenario with this name already exists."

    scenarios[name] = {
        "tickers": tickers,
        "weights": weights,
        "expenses": expenses,
    }
    save_scenarios(scenarios)
    return None


def delete_scenario(scenarios: dict, name: str) -> None:
    """Remove a scenario by name and persist."""
    if name in scenarios:
        del scenarios[name]
        save_scenarios(scenarios)


def export_scenarios(scenarios: dict, names: list[str]) -> str:
    """Export selected scenarios as a JSON string."""
    subset = {k: v for k, v in scenarios.items() if k in names}
    return json.dumps(subset, indent=2)


def import_scenarios(scenarios: dict, json_str: str) -> tuple[int, int]:
    """Import scenarios from JSON string. Returns (imported, skipped) counts."""
    try:
        incoming = json.loads(json_str)
    except json.JSONDecodeError:
        return 0, 0

    imported = 0
    skipped = 0
    for name, details in incoming.items():
        if name in scenarios:
            skipped += 1
            continue
        # Basic validation
        if "tickers" in details and "weights" in details:
            details = ensure_expenses(details)
            scenarios[name] = details
            imported += 1
        else:
            skipped += 1

    if imported > 0:
        save_scenarios(scenarios)
    return imported, skipped
