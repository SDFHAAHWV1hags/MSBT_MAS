#!/usr/bin/env python3
from __future__ import annotations

import inspect
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


_CALLER_HINTS = ("Agent", "Orchestrator", "GroupChat", "Surfer", "Coder")


def _detect_caller() -> Dict[str, str]:
    try:
        for frame_info in inspect.stack():
            self_obj = frame_info.frame.f_locals.get("self")
            if self_obj is None:
                continue
            cls = type(self_obj).__name__
            if cls.startswith("Instrumented"):
                continue
            if any(k in cls for k in _CALLER_HINTS):
                name = getattr(self_obj, "name", None) or cls
                return {"caller_class": cls, "caller_name": str(name)}
    except Exception:
        pass
    return {"caller_class": "unknown", "caller_name": "unknown"}


class LLMMetricsTracker:

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        start_ts: str,
        duration_ms: float,
        usage: Any,
        finish_reason: Optional[str] = None,
        is_error: bool = False,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        caller = _detect_caller()
        prompt = completion = total = None
        if usage is not None:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if total is None and prompt is not None and completion is not None:
                total = prompt + completion
        rec = {
            "seq": len(self.calls) + 1,
            "start_ts": start_ts,
            "duration_ms": round(float(duration_ms), 3),
            **caller,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "finish_reason": finish_reason,
            "is_error": bool(is_error),
            "error": error,
        }
        self.calls.append(rec)
        return rec

    @staticmethod
    def _num(v: Any) -> float:
        return float(v) if isinstance(v, (int, float)) else 0.0

    def summary(self) -> Dict[str, Any]:
        calls = self.calls
        n = len(calls)
        prompt = sum(self._num(c.get("prompt_tokens")) for c in calls)
        completion = sum(self._num(c.get("completion_tokens")) for c in calls)
        total = sum(self._num(c.get("total_tokens")) for c in calls)
        wall = round(sum(self._num(c.get("duration_ms")) for c in calls), 3)
        errors = sum(1 for c in calls if c.get("is_error"))

        by_caller: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            key = c.get("caller_name") or c.get("caller_class") or "unknown"
            b = by_caller.setdefault(key, {
                "num_calls": 0, "llm_wall_ms": 0.0,
                "prompt_tokens": 0.0, "completion_tokens": 0.0,
                "total_tokens": 0.0, "errors": 0,
            })
            b["num_calls"] += 1
            b["llm_wall_ms"] = round(
                b["llm_wall_ms"] + self._num(c.get("duration_ms")), 3)
            b["prompt_tokens"] += self._num(c.get("prompt_tokens"))
            b["completion_tokens"] += self._num(c.get("completion_tokens"))
            b["total_tokens"] += self._num(c.get("total_tokens"))
            b["errors"] += 1 if c.get("is_error") else 0

        return {
            "num_calls": n,
            "num_error_calls": errors,
            "total_llm_wall_ms": wall,
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
            "by_caller": by_caller,
            "calls": calls,
        }


def _usage_of(result: Any) -> Any:
    return getattr(result, "usage", None)


def _finish_reason_of(result: Any) -> Optional[str]:
    fr = getattr(result, "finish_reason", None)
    return str(fr) if fr is not None else None


class InstrumentedModelClient:

    def __init__(self, inner: Any, tracker: Optional[LLMMetricsTracker] = None) -> None:
        self._inner = inner
        self.mas_llm_tracker = tracker or LLMMetricsTracker()

    async def create(self, messages: Any, **kwargs: Any) -> Any:
        start_ts, t0 = _utc_now(), time.perf_counter()
        try:
            result = await self._inner.create(messages, **kwargs)
        except Exception as exc:
            self.mas_llm_tracker.record(
                start_ts=start_ts,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                usage=None, is_error=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.mas_llm_tracker.record(
            start_ts=start_ts,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            usage=_usage_of(result),
            finish_reason=_finish_reason_of(result),
        )
        return result

    async def create_stream(self, messages: Any, **kwargs: Any):
        if kwargs.get("include_usage") is None:
            kwargs["include_usage"] = True
        start_ts, t0 = _utc_now(), time.perf_counter()
        final_result: Any = None
        try:
            async for chunk in self._inner.create_stream(messages, **kwargs):
                if getattr(chunk, "usage", None) is not None:
                    final_result = chunk
                yield chunk
        except Exception as exc:
            self.mas_llm_tracker.record(
                start_ts=start_ts,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                usage=None, is_error=True,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.mas_llm_tracker.record(
            start_ts=start_ts,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            usage=_usage_of(final_result) if final_result is not None else None,
            finish_reason=_finish_reason_of(final_result) if final_result is not None else None,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


try:
    from autogen_core.models import ChatCompletionClient as _ACC

    _ACC.register(InstrumentedModelClient)
except Exception:
    pass
