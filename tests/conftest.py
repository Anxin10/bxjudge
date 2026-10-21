import pytest


@pytest.fixture
def rules():
    """Fresh Rules loaded from config/rules.yaml for every test (safe to mutate)."""
    from bxjudge.rules import load_rules

    return load_rules()


@pytest.fixture
def ref_img():
    """Random-texture 640x360 BGR image."""
    from tests.helpers import make_ref_img

    return make_ref_img()


@pytest.fixture
def th():
    """Thresholds loaded from config/thresholds.yaml."""
    from bxjudge.config import load_thresholds

    return load_thresholds()
