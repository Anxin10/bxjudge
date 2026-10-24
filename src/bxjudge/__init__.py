"""bxjudge: local AI referee for Beyblade X match videos."""

from pathlib import Path

__version__ = "0.1.0"

# Ruling C-X5(c): the repository root, derived from the package location (src/bxjudge), so
# a default config never resolves from the current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTOCALIB_CONFIG = PROJECT_ROOT / "config" / "autocalib.yaml"
