"""Put the vendored streaming detector on sys.path without executing its driver.

A_REDIN.py imports ``QS_VAR`` from ``main``. Importing the real driver pulls a
GUI matplotlib backend and every dataset module. A tiny stand-in module
satisfies that import; the detector classes themselves are unchanged.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ARED = PROJECT_ROOT / "vendor" / "A_RED_INF"
VENDOR_FARPOINT = PROJECT_ROOT / "vendor" / "farpoint"

_BOOTSTRAPPED = False


def bootstrap_ared() -> Path:
    """Ensure vendor/A_RED_INF imports resolve. Safe to call more than once."""
    global _BOOTSTRAPPED
    vendor = str(VENDOR_ARED)
    if not VENDOR_ARED.is_dir():
        raise FileNotFoundError(
            f"Vendored streaming detector not found at {VENDOR_ARED}. "
            "Clone it into vendor/A_RED_INF."
        )
    if vendor not in sys.path:
        sys.path.insert(0, vendor)

    existing = sys.modules.get("main")
    if existing is None:
        stub = types.ModuleType("main")
        stub.QS_VAR = 1
        # The detector imports this name at module scope and then ignores it;
        # every call site passes QS_VAR explicitly.
        sys.modules["main"] = stub
    elif not hasattr(existing, "QS_VAR"):
        existing.QS_VAR = 1

    # FiniteBuffer imports DistanceMetric from sklearn.metrics. Newer
    # scikit-learn builds keep that name under sklearn.metrics.pairwise.
    import sklearn.metrics as metrics

    if not hasattr(metrics, "DistanceMetric"):
        from sklearn.metrics.pairwise import DistanceMetric

        metrics.DistanceMetric = DistanceMetric

    _BOOTSTRAPPED = True
    return VENDOR_ARED


def import_ared_symbols():
    """Return (ARED, Oracle, calculate_single_rel_recall)."""
    bootstrap_ared()
    # more_stats imports pyplot at module scope. Pin a non-interactive backend
    # before that import so headless runs do not require a GUI toolkit.
    import matplotlib

    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    from A_REDIN import ARED
    from Oracle import Oracle
    from more_stats import calculate_single_rel_recall

    return ARED, Oracle, calculate_single_rel_recall
