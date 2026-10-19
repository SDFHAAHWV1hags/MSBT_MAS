#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_SELECTION = {
    "": "adaptive_search_original.py",
    "original": "adaptive_search_original.py",
    "schemea": "adaptive_search_schemeA.py",
    "scheme_a": "adaptive_search_schemeA.py",
    "target3": "adaptive_search_schemeA.py",
}

_key = os.environ.get("MAS_OBJECTIVE_VARIANT", "").strip().lower()
if _key not in _SELECTION:
    raise RuntimeError(
        f"Unknown MAS_OBJECTIVE_VARIANT={_key!r}; expected one of "
        f"{sorted(k for k in _SELECTION if k)} or unset (original)."
    )

_impl_path = Path(__file__).resolve().parent / "objective_variants" / _SELECTION[_key]
_spec = importlib.util.spec_from_file_location(__name__, _impl_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load objective variant implementation: {_impl_path}")
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
