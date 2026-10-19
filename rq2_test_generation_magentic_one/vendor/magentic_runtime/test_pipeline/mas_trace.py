#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adaptive_nsga2 import Gene

_PLAN_MARKERS = ("here is the plan", "plan to follow", "we are working to address")
_REPLAN_MARKERS = ("replan", "revised plan", "new plan", "here is the updated plan")

_PLAN_SECTION_STARTS = (
    "here is the plan to follow", "here is the updated plan to follow",
    "here is the revised plan to follow", "here is the plan", "here is the updated plan",
)
_LEDGER_SECTION_STARTS = (
    "here is an initial fact sheet", "given or verified facts", "facts to look up",
    "facts to derive", "educated guesses", "here is the plan", "here is the updated plan",
)
_CONDITIONAL_PREFIXES = ("if ", "when ", "in case ", "otherwise", "alternatively", "should ")
_CODE_FENCES = ("```sh", "```shell", "```bash", "```python", "```py")


def load_messages(path: Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def content_text(m: Dict[str, Any]) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for item in c:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("arguments") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if c is None else str(c)


def summarize(messages: Sequence[Dict[str, Any]], max_chars: int = 300) -> Dict[str, Any]:
    turns: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        txt = content_text(m)
        turns.append({
            "i": i,
            "type": m.get("type"),
            "source": m.get("source"),
            "chars": len(txt),
            "excerpt": txt[:max_chars],
        })
    return {"num_messages": len(messages), "turns": turns}


def _plan_section(text: str) -> str:
    low = text.lower()
    starts = [low.rfind(m) for m in _PLAN_SECTION_STARTS]
    start = max(starts) if starts else -1
    if start < 0:
        return ""
    seg = text[start:]
    seg_low = seg.lower()
    cut = len(seg)
    for h in _LEDGER_SECTION_STARTS:
        j = seg_low.find(h, 1)
        if 0 <= j < cut:
            cut = j
    return seg[:cut].strip()


def planned_agent_calls(plan_text: str, agents: Sequence[str]) -> List[str]:
    out: List[str] = []
    for line in plan_text.splitlines():
        s = line.strip().lstrip("-*0123456789.) ").strip()
        if not s:
            continue
        if s.lower().startswith(_CONDITIONAL_PREFIXES):
            continue
        for a in agents:
            if a in s and a not in out:
                out.append(a)
    return out


def collect_plan(messages: Sequence[Dict[str, Any]], runtime) -> Dict[str, Any]:
    orch = runtime.orchestrator
    plan_text = ""
    for m in messages:
        if _role(m.get("source"), runtime) != orch:
            continue
        txt = content_text(m)
        if not any(k in txt.lower() for k in _PLAN_MARKERS + _REPLAN_MARKERS):
            continue
        seg = _plan_section(txt)
        if seg:
            plan_text = seg
    return {
        "plan_text": plan_text,
        "planned_agents": planned_agent_calls(plan_text, runtime.agents),
    }


def _rule_id_exact(runtime, event: str) -> Optional[str]:
    for rid, r in runtime.rules.items():
        if r.get("event") == event:
            return rid
    return None


def _rule_id_prefix(runtime, prefix: str) -> Optional[str]:
    for rid, r in runtime.rules.items():
        if str(r.get("event", "")).startswith(prefix):
            return rid
    return None


def _agent_tool_map(runtime) -> Dict[str, str]:
    return {b["agent"]: b["tool"] for b in runtime.structure.get("agent_tool_bindings", [])}


def _role(src: Any, runtime) -> str:
    s = str(src or "")
    if s in runtime.agents:
        return s
    low = s.lower()
    for a in runtime.agents:
        if low == a.lower():
            return a
    if "orchestrator" in low:
        for a in runtime.agents:
            if "orchestrator" in a.lower():
                return a
    return s


_CHAT_MESSAGE_TYPES = (
    "TextMessage", "MultiModalMessage", "ToolCallSummaryMessage", "HandoffMessage", "StopMessage",
)

ROUND_CAP_MARKERS = ("max rounds", "max turns", "round limit", "turn limit")


def is_chat_message(m: Dict[str, Any]) -> bool:
    return str(m.get("type") or "") in _CHAT_MESSAGE_TYPES


def is_round_cap_reason(stop_reason: Optional[str]) -> bool:
    if not stop_reason:
        return False
    low = str(stop_reason).lower()
    return any(k in low for k in ROUND_CAP_MARKERS)


def ledger_rounds(orchestration: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []
    for entry in orchestration or []:
        kind = (entry or {}).get("kind")
        if kind == "progress_ledger":
            led = entry.get("ledger") or {}
            satisfied = bool((led.get("is_request_satisfied") or {}).get("answer"))
            progressed = bool((led.get("is_progress_being_made") or {}).get("answer"))
            in_loop = bool((led.get("is_in_loop") or {}).get("answer"))
            rounds.append({
                "assess": ("ASSESS_COMPLETE" if satisfied
                           else ("ASSESS_STALL" if (not progressed or in_loop)
                                 else "ASSESS_PROGRESS")),
                "delegate": (led.get("next_speaker") or {}).get("answer"),
                "replan": False,
                "complete": satisfied,
            })
        elif kind == "stall_replan" and rounds:
            rounds[-1]["replan"] = True
        elif kind == "task_completed" and rounds:
            rounds[-1]["complete"] = True
    return rounds


def _next_speaker_role(messages: Sequence[Dict[str, Any]], i: int, runtime,
                       exclude: str) -> Optional[str]:
    for j in range(i + 1, len(messages)):
        m = messages[j]
        if not is_chat_message(m):
            continue
        cand = _role(m.get("source"), runtime)
        if cand and cand != exclude:
            return cand
    return None


def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _tool_calls_by_agent(tool_calls: Optional[Sequence[Dict[str, Any]]], runtime
                         ) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in tool_calls or []:
        agent = _role(rec.get("agent"), runtime)
        grouped.setdefault(agent, []).append(dict(rec))
    return grouped


def _take_tool_calls(bucket: Sequence[Dict[str, Any]], before_ts: Optional[datetime]
                     ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    taken: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for rec in bucket:
        ts = _parse_ts(rec.get("ts"))
        if before_ts is None or ts is None or ts.tzinfo is None or ts <= before_ts:
            taken.append(rec)
        else:
            rest.append(rec)
    return taken, rest


def derive_candidates(messages: Sequence[Dict[str, Any]], runtime,
                      orchestration: Optional[Sequence[Dict[str, Any]]] = None,
                      stop_reason: Optional[str] = None,
                      tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
                      ) -> List[Tuple[Gene, Dict[str, Any]]]:
    agents = list(runtime.agents)
    orch = runtime.orchestrator
    atool = _agent_tool_map(runtime)
    rounds = ledger_rounds(orchestration)
    pending = _tool_calls_by_agent(tool_calls, runtime)
    recorded_agents = set(pending)
    out: List[Tuple[Gene, Dict[str, Any]]] = []

    def add(prefix: str, event: str, params: Dict[str, str], idx: int, src: str,
            why: str) -> None:
        rid = _rule_id_exact(runtime, event) or _rule_id_prefix(runtime, prefix)
        if not rid:
            return
        out.append((
            Gene(rid, event, tuple(sorted(params.items()))),
            {"msg_index": idx, "source": src, "why": why},
        ))

    planned = False
    ri = 0
    last_chat_src: Optional[str] = None
    delegated: List[str] = []

    for i, m in enumerate(messages):
        src = _role(m.get("source"), runtime)
        txt = content_text(m)
        low = txt.lower()
        mtype = str(m.get("type") or "")

        if src == "user" and i == 0:
            add("RECEIVE", "RECEIVE", {}, i, src, "user issues the task")
            continue

        if "ToolCallRequest" in mtype:
            ag = src if src in agents else None
            if ag:
                tool = atool.get(ag)
                if tool:
                    add("TOOL_CALL", f"TOOL_CALL({ag},{tool})", {"agent": ag, "tool": tool},
                        i, src, f"{ag} initiates tool call ({mtype})")
            continue
        if "ToolCallExecution" in mtype:
            ag = src if src in agents else None
            if ag:
                tool = atool.get(ag)
                if tool:
                    add("TOOL_RETURN", f"TOOL_RETURN({tool},{ag})", {"agent": ag, "tool": tool},
                        i, src, f"{ag} receives tool return ({mtype})")
            continue

        if not is_chat_message(m):
            continue

        if src == orch:
            last_chat_src = None
            if _plan_section(txt):
                if not planned:
                    add("PLAN", "PLAN", {}, i, src, "orchestrator gives initial plan (task ledger)")
                    planned = True
                    continue
                if ri < len(rounds) and rounds[ri]["replan"]:
                    add(rounds[ri]["assess"], rounds[ri]["assess"], {}, i, src,
                        "progress ledger judges no progress this round; replanning triggered")
                    add("REPLAN", "REPLAN", {}, i, src, "orchestrator replans (stall reached threshold)")
                    ri += 1
                else:
                    add("REPLAN", "REPLAN", {}, i, src, "orchestrator gives updated plan")
                add("PLAN", "PLAN", {}, i, src, "orchestrator gives the (updated) plan")
                continue

            nxt_src = _next_speaker_role(messages, i, runtime, exclude=orch)

            if nxt_src in agents:
                target = nxt_src
                if ri < len(rounds):
                    r = rounds[ri]
                    add(r["assess"], r["assess"], {}, i, src,
                        "progress ledger judgement this round (orchestrator)")
                    if r["delegate"] in agents:
                        target = r["delegate"]
                    ri += 1
                add("DELEGATE", f"DELEGATE({target})", {"agent": target}, i, src,
                    f"orchestrator delegates to {target}")
                delegated.append(target)
                if (target == "ComputerTerminal" and "ComputerTerminal" not in recorded_agents
                        and any(f in low for f in _CODE_FENCES)):
                    tool = atool.get("ComputerTerminal")
                    if tool:
                        add("TOOL_CALL", f"TOOL_CALL(ComputerTerminal,{tool})",
                            {"agent": "ComputerTerminal", "tool": tool}, i, src,
                            "orchestrator issues executable code block; ComputerTerminal calls Shell (text heuristic)")
                        add("TOOL_RETURN", f"TOOL_RETURN({tool},ComputerTerminal)",
                            {"agent": "ComputerTerminal", "tool": tool}, i + 1, "ComputerTerminal",
                            "ComputerTerminal returns execution result")
                continue

            if is_round_cap_reason(stop_reason):
                add("BUDGET_TERMINATE", "BUDGET_TERMINATE", {}, i, src,
                    f"team forced to finalize due to turn limit: stop_reason={stop_reason!r}")
            elif ri < len(rounds):
                r = rounds[ri]
                add("ASSESS_COMPLETE", "ASSESS_COMPLETE", {}, i, src,
                    "progress ledger judges request fulfilled this round")
                add("FINALIZE", "FINALIZE", {}, i, src, "orchestrator outputs final answer and finishes")
                ri += 1
            elif rounds and not rounds[-1]["complete"]:
                add("BUDGET_TERMINATE", "BUDGET_TERMINATE", {}, i, src,
                    f"orchestrator finalized without ever judging the request fulfilled: stop_reason={stop_reason!r}")
            else:
                add("ASSESS_COMPLETE", "ASSESS_COMPLETE", {}, i, src, "orchestrator judges task complete")
                add("FINALIZE", "FINALIZE", {}, i, src, "orchestrator outputs final answer and finishes")
            continue

        if src in agents:
            if last_chat_src == src:
                continue
            last_chat_src = src
            if _next_speaker_role(messages, i, runtime, exclude=src) == orch:
                drained, pending[src] = _take_tool_calls(
                    pending.get(src, []), _parse_ts(m.get("created_at")))
                tool = atool.get(src)
                for rec in drained:
                    if not tool:
                        break
                    add("TOOL_CALL", f"TOOL_CALL({src},{tool})", {"agent": src, "tool": tool},
                        i, src,
                        f"{src} calls {tool} ({rec.get('action')}, ground truth from {rec.get('origin')})")
                    add("TOOL_RETURN", f"TOOL_RETURN({tool},{src})", {"agent": src, "tool": tool},
                        i, src, f"{tool} returns result to {src}")
                add("REPORT", f"REPORT({src})", {"agent": src}, i, src, f"{src} reports to orchestrator")
            continue

    return out


def repair(candidates: Sequence[Tuple[Gene, Dict[str, Any]]], runtime):
    state = json.loads(json.dumps(runtime.initial_state))
    kept: List[Tuple[Gene, Dict[str, Any]]] = []
    dropped: List[Dict[str, Any]] = []
    for gene, ev in candidates:
        enabled = {g.key for g in runtime.enabled_genes(state)}
        if gene.key in enabled:
            state = runtime.apply(state, gene)
            kept.append((gene, ev))
        else:
            dropped.append({
                "gene": gene.to_dict(),
                "reason": (f"this transition is not enabled in the current state"
                           f"（activeEntity={state.get('activeEntity')}, "
                           f"phase={state.get('phase')}, budget={state.get('budget')}）"),
                "evidence": ev,
            })
    return kept, dropped, state


def trace_to_path(messages: Sequence[Dict[str, Any]], runtime,
                  outcome: Optional[str] = None,
                  orchestration: Optional[Sequence[Dict[str, Any]]] = None,
                  stop_reason: Optional[str] = None,
                  tool_calls: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    candidates = derive_candidates(messages, runtime, orchestration=orchestration,
                                   stop_reason=stop_reason, tool_calls=tool_calls)
    kept, dropped, final_state = repair(candidates, runtime)

    genes = [g for g, _ in kept]
    evidence = [
        {"order": n, **ev, "gene": g.to_dict()}
        for n, (g, ev) in enumerate(kept)
    ]

    simulated_ok, sim_state, _ = runtime.simulate(genes)
    terminal = runtime.is_terminal(final_state)
    rounds = ledger_rounds(orchestration)

    return {
        "converted_at": datetime.now().isoformat(timespec="seconds"),
        "outcome": outcome,
        "num_candidates": len(candidates),
        "num_genes": len(genes),
        "num_dropped": len(dropped),
        "legal": bool(simulated_ok),
        "reached_terminal": bool(terminal),
        "termination": (
            "FINALIZE" if any(g.event == "FINALIZE" for g in genes)
            else ("BUDGET_TERMINATE" if runtime.is_terminal(final_state) else None)
        ),
        "termination_reason": next(
            (ev["why"] for g, ev in kept if g.event in ("FINALIZE", "BUDGET_TERMINATE")), None),
        "stop_reason": stop_reason,
        "num_ledger_rounds": len(rounds),
        "num_ledger_replans": sum(1 for r in rounds if r["replan"]),
        "num_tool_call_records": len(tool_calls or []),
        "num_tool_calls": sum(1 for g in genes if g.event.startswith("TOOL_CALL(")),
        "final_state": final_state,
        "sim_state_matches": sim_state == final_state,
        "path_keys": [g.key for g in genes],
        "events": [g.event for g in genes],
        "delegated_agents": sorted({g.event[len("DELEGATE("):-1]
                                    for g in genes if g.event.startswith("DELEGATE(")
                                    and g.event.endswith(")")}),
        "genes": [g.to_dict() for g in genes],
        "evidence": evidence,
        "dropped": dropped,
    }


def event_keys(path_keys: Sequence[Any]) -> List[str]:
    out = []
    for k in path_keys:
        _rid, sep, ev = str(k).partition("::")
        out.append(ev if sep else str(k))
    return out


def save_path(result: Dict[str, Any], out_path: Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def genes_from_result(result: Dict[str, Any]) -> List[Gene]:
    return [
        Gene(g["rule_id"], g["event"],
             tuple(sorted((str(k), str(v)) for k, v in (g.get("params") or {}).items())))
        for g in result.get("genes", [])
    ]
