#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adaptive_nsga2 import Gene
from mas_trace import repair, save_path


def _rule_id(runtime, event: str) -> Optional[str]:
    for rid, r in runtime.rules.items():
        if r.get("event") == event:
            return rid
    prefix = event.split("(", 1)[0]
    for rid, r in runtime.rules.items():
        if str(r.get("event", "")).startswith(prefix + "("):
            return rid
    return None


def derive_candidates(messages: Sequence[Dict[str, Any]], runtime,
                      orchestration: Optional[Dict[str, Any]] = None,
                      outcome: Optional[str] = None) -> List[Tuple[Gene, Dict[str, Any]]]:
    orchestration = orchestration or {}
    func_map: Dict[str, str] = orchestration.get("function_to_model_tool") or {}
    bindings = set(runtime.bindings)
    budget = int(orchestration.get("budget") or runtime.initial_state.get("budget") or 0)
    agents = set(runtime.agents)

    candidates: List[Tuple[Gene, Dict[str, Any]]] = []

    def add(event: str, params: Dict[str, str], idx: int, src: str, why: str) -> None:
        rid = _rule_id(runtime, event)
        if not rid:
            return
        candidates.append((
            Gene(rid, event, tuple(sorted(params.items()))),
            {"msg_index": idx, "source": src, "why": why},
        ))

    pending: Dict[str, Tuple[str, str, str, Any, int]] = {}

    def emit_call_return(call_id: str, idx: int, is_error: Any, phase: str) -> None:
        info = pending.pop(call_id, None)
        if info is None:
            return
        agent, model_tool, func_name, arguments, req_idx = info
        if (agent, model_tool) not in bindings:
            add(f"TOOL_CALL({agent},{model_tool})", {"agent": agent, "tool": model_tool},
                req_idx, agent, f"[skip] agent {agent} has no bound tool {model_tool} (function={func_name})")
            return
        add(f"TOOL_CALL({agent},{model_tool})", {"agent": agent, "tool": model_tool},
            req_idx, agent,
            f"{agent} calls {model_tool} (function={func_name}, arguments={arguments})")
        add(f"TOOL_RETURN({model_tool},{agent})", {"agent": agent, "tool": model_tool},
            idx, agent,
            f"{model_tool} returns result to {agent}"
            + ("(tool execution failed, is_error=True)" if is_error else ""))

    n_selections = 0
    for i, m in enumerate(messages):
        mtype = str(m.get("type") or "")
        src = str(m.get("source") or "")
        content = m.get("content")

        if mtype == "SelectSpeakerEvent":
            chosen = None
            if isinstance(content, list) and content:
                chosen = str(content[0])
            if not chosen:
                continue
            if n_selections == 0:
                if chosen in agents:
                    add(f"SELECT_INITIAL({chosen})", {"agent": chosen}, i, src,
                        "manager selects the first speaker (SELECT_INITIAL)")
            else:
                if chosen in agents:
                    add(f"SELECT_NEXT({chosen})", {"agent": chosen}, i, src,
                        f"manager selects speaker #{n_selections + 1} (SELECT_NEXT)")
            n_selections += 1
            continue

        if mtype == "ToolCallRequestEvent":
            agent = src if src in agents else None
            if not agent or not isinstance(content, list):
                continue
            for fc in content:
                fname = str(fc.get("name") or "")
                model_tool = func_map.get(fname)
                call_id = str(fc.get("id") or "")
                arguments = fc.get("arguments")
                if not model_tool or not call_id:
                    continue
                pending[call_id] = (agent, model_tool, fname, arguments, i)
            continue

        if mtype == "ToolCallExecutionEvent":
            if not isinstance(content, list):
                continue
            for result in content:
                call_id = str(result.get("call_id") or "")
                is_error = bool(result.get("is_error"))
                if call_id in pending:
                    emit_call_return(call_id, i, is_error, "return")
            continue

    for call_id, (agent, model_tool, func_name, arguments, req_idx) in list(pending.items()):
        if (agent, model_tool) not in bindings:
            continue
        add(f"TOOL_CALL({agent},{model_tool})", {"agent": agent, "tool": model_tool},
            req_idx, agent,
            f"{agent} calls {model_tool} (function={func_name}) but no return event observed")

    final_idx = len(messages) - 1
    if outcome == "COMPLETED":
        add("TERMINATE", {}, final_idx, "captain_exec", "expert outputs TERMINATE, task completed normally (CA-R5)")
    elif outcome == "BUDGET_EXHAUSTED":
        add("BUDGET_TERMINATE", {}, final_idx, "captain_exec",
            f"selections reached model budget {budget} without completion (CA-R6)")
    return candidates


def trace_to_path(messages: Sequence[Dict[str, Any]], runtime,
                  outcome: Optional[str] = None,
                  orchestration: Optional[Dict[str, Any]] = None,
                  stop_reason: Optional[str] = None,
                  tool_calls: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    candidates = derive_candidates(messages, runtime, orchestration=orchestration, outcome=outcome)
    kept, dropped, final_state = repair(candidates, runtime)

    genes = [g for g, _ in kept]
    evidence = [{"order": n, **ev, "gene": g.to_dict()}
                for n, (g, ev) in enumerate(kept)]

    simulated_ok, sim_state, _ = runtime.simulate(genes)
    terminal = runtime.is_terminal(final_state)
    selections = (orchestration or {}).get("selections") or []

    request_records = [r for r in (tool_calls or []) if not r.get("phase") == "return"]
    return_records = [r for r in (tool_calls or []) if r.get("phase") == "return"]

    return {
        "converted_at": datetime.now().isoformat(timespec="seconds"),
        "outcome": outcome,
        "num_candidates": len(candidates),
        "num_genes": len(genes),
        "num_dropped": len(dropped),
        "legal": bool(simulated_ok),
        "reached_terminal": bool(terminal),
        "termination": (
            "TERMINATE" if any(g.event == "TERMINATE" for g in genes)
            else ("BUDGET_TERMINATE" if any(g.event == "BUDGET_TERMINATE" for g in genes) else None)
        ),
        "termination_reason": next(
            (ev["why"] for g, ev in kept if g.event in ("TERMINATE", "BUDGET_TERMINATE")), None),
        "stop_reason": stop_reason,
        "num_selections": len(selections),
        "num_tool_call_records": len(request_records),
        "num_tool_return_records": len(return_records),
        "num_tool_calls": sum(1 for g in genes if g.event.startswith("TOOL_CALL(")),
        "final_state": final_state,
        "sim_state_matches": sim_state == final_state,
        "path_keys": [g.key for g in genes],
        "events": [g.event for g in genes],
        "selected_agents": [s.get("speaker") for s in selections if s.get("speaker")],
        "delegated_agents": sorted({
            g.event[len("SELECT_NEXT("):-1]
            for g in genes if g.event.startswith("SELECT_NEXT(") and g.event.endswith(")")
        }),
        "genes": [g.to_dict() for g in genes],
        "evidence": evidence,
        "dropped": dropped,
    }


def genes_from_result(result: Dict[str, Any]) -> List[Gene]:
    return [
        Gene(g["rule_id"], g["event"],
             tuple(sorted((str(k), str(v)) for k, v in (g.get("params") or {}).items())))
        for g in result.get("genes", [])
    ]
