"""Experiment wrapper: run bxjudge.cli with E9 (later stadium windows) switched on."""
import sys
from dataclasses import replace
from bxjudge import stadium_locator, stadium_finder
_orig = stadium_finder.load_stadium_finder_config
def _on(*a, **k):
    return replace(_orig(*a, **k), later_windows_enabled=True)
stadium_locator.load_stadium_finder_config = _on
from bxjudge.cli import main
sys.exit(main())
