#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List


def apply_leave_current_query_out(concretizer, model: Dict[str, Any]) -> List[str]:
    return concretizer.exclude_current_model_source(model)
