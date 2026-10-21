import pytest


@pytest.fixture
def rules():
    """Fresh Rules loaded from config/rules.yaml for every test (safe to mutate)."""
    from bxjudge.rules import load_rules

    return load_rules()
