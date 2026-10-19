#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from brs_user_task_generator import BRSConcretizer


def default_library_root() -> Path:
    env = os.environ.get("BRS_LIBRARY_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "library",
        here.parent.parent / "brs_v0_4" / "library",
        Path.cwd() / "brs_v0_4" / "library",
        Path.cwd() / "library",
    ]
    for p in candidates:
        if (p / "task_schema_fragment_library.v0_4.json").exists():
            return p
    raise FileNotFoundError(
        "BRS v0.4 library not found. Set BRS_LIBRARY_ROOT or pass "
        "library_root explicitly (need task_schema_fragment_library.v0_4.json)."
    )


def make_concretizer(
    chat: Any,
    temperature: float,
    max_tokens: int,
    *,
    library_root: Optional[Path] = None,
    grounding_mode: str = "bootstrap",
    signature_policy: str = "exact",
    n_surface_candidates: int = 3,
    allow_source_task_fallback: bool = True,
    attachment_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> BRSConcretizer:
    return BRSConcretizer(
        library_root or default_library_root(),
        chat=chat,
        temperature=temperature,
        max_tokens=max_tokens,
        n_surface_candidates=n_surface_candidates,
        grounding_mode=grounding_mode,
        signature_policy=signature_policy,
        allow_source_task_fallback=allow_source_task_fallback,
        attachment_map=attachment_map,
    )
