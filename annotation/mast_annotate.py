#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


FM_DEFINITIONS: list[tuple[str, str, str]] = [
    ("FM-1.1", "Disobey task specification",
     "Failure to adhere to the specified constraints or requirements of a given task, "
     "leading to suboptimal or incorrect outcomes."),
    ("FM-1.2", "Disobey role specification",
     "Failure to adhere to the defined responsibilities and constraints of an assigned "
     "role, potentially leading to an agent behaving like another."),
    ("FM-1.3", "Step repetition",
     "Unnecessary reiteration of previously completed steps in a process, potentially "
     "causing delays or errors in task completion."),
    ("FM-1.4", "Loss of conversation history",
     "Unexpected context truncation, disregarding recent interaction history and "
     "reverting to an antecedent conversational state."),
    ("FM-1.5", "Unaware of termination conditions",
     "Lack of recognition or understanding of the criteria that should trigger the "
     "termination of the agents' interaction, potentially leading to unnecessary "
     "continuation."),
    ("FM-2.1", "Conversation reset",
     "Unexpected or unwarranted restarting of a dialogue, potentially losing context "
     "and progress made in the interaction."),
    ("FM-2.2", "Fail to ask for clarification",
     "Inability to request additional information when faced with unclear or incomplete "
     "data, potentially resulting in incorrect actions."),
    ("FM-2.3", "Task derailment",
     "Deviation from the intended objective or focus of a given task, potentially "
     "resulting in irrelevant or unproductive actions."),
    ("FM-2.4", "Information withholding",
     "Failure to share or communicate important data or insights that an agent possesses "
     "and could impact decision-making of other agents if shared."),
    ("FM-2.5", "Ignored other agent's input",
     "Disregarding or failing to adequately consider input or recommendations provided "
     "by other agents in the system, potentially leading to suboptimal decisions or "
     "missed opportunities for collaboration."),
    ("FM-2.6", "Reasoning-action mismatch",
     "Discrepancy between the logical reasoning process and the actual actions taken by "
     "the agent, potentially resulting in unexpected or undesired behaviors."),
    ("FM-3.1", "Premature termination",
     "Ending a dialogue, interaction or task before all necessary information has been "
     "exchanged or objectives have been met, potentially resulting in incomplete or "
     "incorrect outcomes."),
    ("FM-3.2", "No or incomplete verification",
     "(Partial) omission of proper checking or confirmation of task outcomes or system "
     "outputs, potentially allowing errors or inconsistencies to propagate undetected."),
    ("FM-3.3", "Incorrect verification",
     "Failure to adequately validate or cross-check crucial information or decisions "
     "during the iterations, potentially leading to errors or vulnerabilities in the "
     "system."),
]
FM_CODES = [c for c, _, _ in FM_DEFINITIONS]

ENV_BLOCKER_TYPES = [
    "captcha", "http_blocked_or_forbidden", "api_billing_or_quota",
    "browser_or_tool_crash", "network_or_dns", "sandbox_or_permission",
    "wallclock_timeout", "other_environment",
]

PER_MSG_CAP = {
    "user": 6000,
    "MagenticOneOrchestrator": 2200,
    "Coder": 2600,
    "WebSurfer": 1100,
    "FileSurfer": 1100,
    "ComputerTerminal": 1100,
}
PER_MSG_CAP_DEFAULT = 1100
DIGEST_CHARS = 180
HEAD_KEEP = 4
TAIL_KEEP = 10
DIALOGUE_CHAR_BUDGET = 90000
LEDGER_REASON_CAP = 700
FINAL_ANSWER_CAP = 1500

PROMPT_VERSION = "mast-v4"


def _is_coordinator(source: str) -> bool:
    s = (source or "").lower()
    return "orchestrator" in s or "manager" in s


def resolve_model_path(request: dict[str, Any], request_path: Path) -> Path:
    suite = Path(__file__).resolve().parent.parent
    raw = request.get("model_path") or ""
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append((request_path.parent / p).resolve())
        candidates.append(suite / "models" / "selected" / Path(raw).name)
    candidates.append(suite / "models" / "selected" / "Magentic-One.json")
    for c in candidates:
        if c and c.exists():
            return c
    raise FileNotFoundError(f"model JSON not found for model_path={raw!r}")


def load_env_key() -> None:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",
        Path("/home/zs/mas/.env"),
    ]
    for p in candidates:
        if p.is_file():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "DEEPSEEK_API_KEY":
                    os.environ["DEEPSEEK_API_KEY"] = v.strip().strip('"').strip("'")
                    return


def load_llm_profile(config_path: Path, profile: str) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    prof = cfg["llms"][profile]
    key_env = prof.get("api_key_env", "DEEPSEEK_API_KEY")
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"API key env {key_env} is empty")
    return {
        "base_url": prof["base_url"],
        "model": prof["model"],
        "api_key": key,
        "timeout_sec": int(prof.get("timeout_sec", 180)),
    }



def _short(s: Any, cap: int) -> str:
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False, default=str)
    if len(s) <= cap:
        return s
    return s[:cap] + f" ...[truncated {len(s)-cap} chars]"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def _role_from_system_prompt(sp: str) -> str:
    if not sp:
        return ""
    for line in sp.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("## your role"):
            continue
        return line.lstrip("- ").strip()[:300]
    return sp.strip().splitlines()[0][:300]


def build_agent_profile(model_json_path: Path) -> str:
    m = json.loads(model_json_path.read_text(encoding="utf-8"))
    lines = [f"System under test: {m.get('model_id')} ({m.get('framework')})", "",
             "Agents and their assigned roles:"]
    for name, spec in m.get("agent_definitions", {}).items():
        role = spec.get("role") or _role_from_system_prompt(spec.get("system_prompt", ""))
        if role.startswith(f"{name}:"):
            role = role[len(name) + 1:].strip()
        lines.append(f"- {name}: {role}")

    struct = m.get("structure", {}) or {}
    coord = struct.get("agent_coordination", {}) or {}
    selector = coord.get("selector")
    if selector:
        tools = struct.get("tools", [])
        lines += [
            "",
            "Team architecture (from the test model):",
            f"- Selector group-chat: after every message a coordinator ({selector}) "
            "selects the next speaker from ALL agents; the same agent may be selected "
            "again consecutively, and agents can request a peer by name. There is no "
            "central task ledger - progress judgments appear in the coordinator's / "
            "agents' own messages (Section 5).",
            f"- Tools declared by the model: {', '.join(tools) if tools else 'none'}. "
            "The runtime agent-to-tool bindings and the harness-injected runtime "
            "protocol are shown verbatim in Section 3.",
            "- Termination is supposed to happen only when the task is actually "
            "complete and the final-answer termination token is emitted (or the turn "
            "budget is exhausted).",
        ]
        return "\n".join(lines)

    cfg = m.get("configuration_parameters", {})
    lines += [
        "",
        "Coordination semantics (from the test model):",
        "- Strict star topology: the Orchestrator is the ONLY agent that selects and "
        "delegates subtasks to the specialists; specialists only report back to "
        "the Orchestrator. They cannot talk to each other.",
        f"- Max conversation rounds budget: {cfg.get('max_rounds')}.",
        f"- Stall policy: stall count {cfg.get('stall_policy')}.",
        "- The Orchestrator maintains a task ledger and, after every specialist report, "
        "answers three progress questions: is_request_satisfied / is_progress_being_made "
        "/ is_in_loop; it also records pending questions and the next speaker.",
        "- Termination is supposed to happen only when is_request_satisfied is judged "
        "true (or the round budget is exhausted).",
    ]
    return "\n".join(lines)


def _extract_case_info(request: dict[str, Any], ck: dict[str, Any]) -> dict[str, Any]:
    lr = ck.get("last_round", {})
    query = lr.get("testcase", {}).get("query") or lr.get("query") or request.get("query", "")
    info: dict[str, Any] = {
        "query": query,
        "outcome": lr.get("outcome"),
        "failure_signature": lr.get("failure_signature"),
        "final_answer": _short(lr.get("final_answer"), FINAL_ANSWER_CAP),
    }
    timings = lr.get("timings") or {}
    if isinstance(timings, dict):
        for k in ("execution_sec", "wall_sec", "total_sec"):
            if k in timings:
                info[k] = timings[k]
    return info


def format_case_info(info: dict[str, Any]) -> str:
    lines = ["TASK QUERY:", info["query"], "",
             f"RUN OUTCOME: {info.get('outcome')}"]
    if info.get("failure_signature"):
        lines.append(f"FAILURE SIGNATURE: {_short(info['failure_signature'], 500)}")
    for k in ("execution_sec", "wall_sec", "total_sec"):
        if k in info:
            lines.append(f"{k}: {info[k]}")
    lines += ["", "FINAL ANSWER SUBMITTED BY THE MAS:", str(info.get("final_answer")),
              "", "NOTE: No answer key is provided. Judge success and constraint "
                  "compliance ONLY against the literal wording of TASK QUERY above."]
    return "\n".join(lines)


def format_ledger(orch: dict[str, Any]) -> str:
    entries = orch.get("entries", [])
    lines = [f"max_turns={orch.get('max_turns')} stop_reason={orch.get('stop_reason')!r} "
             f"ledger_entries={len(entries)}",
             "Full Orchestrator progress-ledger sequence (chronological):"]
    seq = 0
    for e in entries:
        if e.get("kind") != "progress_ledger":
            lines.append(f"[{seq}] <{e.get('kind')}>")
            seq += 1
            continue
        led = e.get("ledger", {})
        parts = []
        for key, label in (("is_request_satisfied", "satisfied"),
                           ("is_progress_being_made", "progress"),
                           ("is_in_loop", "loop")):
            v = led.get(key)
            if isinstance(v, dict):
                ans = v.get("answer")
                reason = _short(v.get("reason"), LEDGER_REASON_CAP)
                parts.append(f"{label}={ans} ({reason})")
            elif v is not None:
                parts.append(f"{label}={v}")
        qd = led.get("questions_or_decisions_for_next_speaker")
        if qd:
            parts.append("pending: " + _short(qd, 200))
        ns = led.get("next_speaker")
        if ns:
            parts.append(f"next_speaker={ns}")
        lines.append(f"[{seq}] " + " | ".join(parts))
        seq += 1
    return "\n".join(lines)


def format_selector_orchestration(orch: dict[str, Any]) -> str:
    sels = orch.get("selections", []) or []
    seq = [s.get("speaker") for s in sels]
    from collections import Counter
    counts = Counter(seq)
    repeats = [f"{a}x{n}" for a, n in counts.items()]
    consec = []
    run_a, run_n = None, 0
    for a in seq + [None]:
        if a == run_a:
            run_n += 1
        else:
            if run_a and run_n >= 2:
                consec.append(f"{run_a} spoke {run_n} times in a row")
            run_a, run_n = a, 1
    lines = [
        f"family={orch.get('family')} model={orch.get('model_id')} "
        f"budget={orch.get('budget')} max_turns={orch.get('max_turns')} "
        f"num_selections={orch.get('num_selections')}",
        f"stop_reason={orch.get('stop_reason')!r}",
        f"outcome={orch.get('outcome')!r}",
        "Agent->tool bindings: "
        + (", ".join(f"{b.get('agent')} -> {b.get('tool')}"
                     for b in orch.get("bindings", [])) or "(none)"),
        "",
        "Harness-injected runtime protocol (verbatim):",
        str(orch.get("runtime_protocol_header", "")).strip() or "(not recorded)",
        "",
        f"Manager speaker selections ({len(seq)}): " + " -> ".join(str(s) for s in seq),
        "Speaker counts: " + ", ".join(repeats),
    ]
    if consec:
        lines.append("CONSECUTIVE RE-SELECTIONS (potential repetition/stall): "
                     + "; ".join(consec))
    return "\n".join(lines)


def format_orchestration(orch: dict[str, Any]) -> str:
    if isinstance(orch, dict) and "selections" in orch and "entries" not in orch:
        return format_selector_orchestration(orch)
    return format_ledger(orch)


def format_tool_calls(tc: dict[str, Any]) -> str:
    entries = tc.get("entries", []) if isinstance(tc, dict) else tc
    lines = [f"{len(entries)} tool calls (chronological):"]
    for e in entries:
        err = " ERROR" if e.get("is_error") else ""
        url = f" url={e['url']}" if e.get("url") else ""
        args = _short(e.get("arguments"), 90).replace("\n", " ")
        dur = e.get("duration_ms")
        name = e.get("action") or e.get("function") or "?"
        tool = f"/{e['tool']}" if e.get("tool") else ""
        lines.append(f"#{e.get('seq')} [{e.get('agent')}] {name}{tool}{err} "
                     f"{dur}ms{url} args={args}")
    return "\n".join(lines)


def build_dialogue(messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    norm: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        src = m.get("source", "?")
        text = _message_text(m.get("content"))
        if src == "user":
            cap = PER_MSG_CAP["user"]
        elif _is_coordinator(src):
            cap = 2200
        else:
            cap = PER_MSG_CAP.get(src, PER_MSG_CAP_DEFAULT)
        truncated = len(text) > cap
        norm.append({"idx": i, "source": src, "text": _short(text, cap),
                     "truncated": truncated, "full_len": len(text)})

    total = sum(len(x["text"]) for x in norm)
    meta = {"n_messages": len(messages), "n_fully_kept": len(norm),
            "n_sampled_out": 0, "per_message_truncated": 0}
    if total <= DIALOGUE_CHAR_BUDGET:
        meta["per_message_truncated"] = sum(1 for x in norm if x["truncated"])
        return _render_messages(norm, omitted={}), meta

    n = len(norm)
    keep: set[int] = set(range(min(HEAD_KEEP, n))) | set(range(max(0, n - TAIL_KEEP), n))
    for x in norm:
        if _is_coordinator(x["source"]) or x["source"] == "Coder":
            keep.add(x["idx"])
    remaining_budget = DIALOGUE_CHAR_BUDGET - sum(len(norm[i]["text"]) for i in keep)
    candidates = [x for x in norm if x["idx"] not in keep]
    spread = list(candidates)
    cur = 0
    while remaining_budget > 0 and cur < len(spread):
        step = max(1, len(spread) // max(1, min(40, len(spread))))
        added = False
        for x in spread[::step]:
            if x["idx"] in keep:
                continue
            if len(x["text"]) <= remaining_budget:
                keep.add(x["idx"])
                remaining_budget -= len(x["text"])
                added = True
        cur += 1
        if not added:
            break

    omitted = {x["idx"]: x for x in norm if x["idx"] not in keep}
    meta["n_fully_kept"] = len(keep)
    meta["n_sampled_out"] = len(omitted)
    meta["per_message_truncated"] = sum(1 for x in norm if x["truncated"])
    return _render_messages([x for x in norm if x["idx"] in keep], omitted), meta


def _render_messages(kept: list[dict[str, Any]], omitted: dict[int, dict]) -> str:
    kept_set = {x["idx"] for x in kept}
    lines: list[str] = []
    pending_digest: list[dict[str, Any]] = []

    def flush_digest() -> None:
        if not pending_digest:
            return
        idxs = [x["idx"] for x in pending_digest]
        lines.append(f"    ---- {len(pending_digest)} message(s) {idxs[0]}..{idxs[-1]} "
                     f"removed by token-budget sampling; short gists below ----")
        for x in pending_digest:
            gist = x["text"].replace("\n", " ")[:DIGEST_CHARS]
            lines.append(f"    [msg {x['idx']} {x['source']} (gist)]: {gist}")
        lines.append("    ---- end of sampled-out gists ----")
        pending_digest.clear()

    all_msgs = sorted(kept + list(omitted.values()), key=lambda x: x["idx"])
    for x in all_msgs:
        if x["idx"] not in kept_set:
            pending_digest.append(x)
            continue
        flush_digest()
        tag = " [truncated]" if x["truncated"] else ""
        lines.append(f"[msg {x['idx']}] {x['source']}{tag}:")
        lines.append(x["text"])
        lines.append("")
    flush_digest()
    return "\n".join(lines)


def reconstruct_messages_from_framework(fw_path: Path) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    with fw_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            i = line.find('{"payload"')
            if i < 0:
                continue
            try:
                outer = json.loads(line[i:])
                payload = json.loads(outer["payload"])
            except (json.JSONDecodeError, KeyError):
                continue
            cm = ((payload.get("response") or {}).get("chat_message")
                  or payload.get("message"))
            if not (isinstance(cm, dict) and isinstance(cm.get("content"), str)
                    and cm.get("source")):
                continue
            src = cm["source"]
            if src.endswith("Event") or src in ("ThoughtEvent",):
                continue
            raw.append({"source": src, "content": cm["content"]})

    deduped: list[dict[str, Any]] = []
    for m in raw:
        if deduped and deduped[-1]["source"] == m["source"] \
                and deduped[-1]["content"] == m["content"]:
            continue
        deduped.append(m)
    return [{"idx": i, "source": m["source"], "content": m["content"]}
            for i, m in enumerate(deduped)]


def build_evidence_pack_from_framework(logs_dir: Path, request_path: Path,
                                       checkpoint_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    model_path = resolve_model_path(request, request_path)

    messages = reconstruct_messages_from_framework(logs_dir / "framework.log")
    dialogue, dmeta = build_dialogue(messages)
    info = _extract_case_info(request, ck)

    sections = [
        "=" * 70,
        "SECTION 1 - SYSTEM UNDER TEST (agent roles & coordination rules)",
        "=" * 70,
        build_agent_profile(model_path),
        "", "=" * 70,
        "SECTION 2 - TASK, OUTCOME AND ORACLE",
        "=" * 70,
        format_case_info(info),
        "", "=" * 70,
        "SECTION 3 - PROGRESS LEDGER",
        "=" * 70,
        "NOTE: This run was killed by the wallclock timeout before structured "
        "orchestration logs were flushed, so no parsed ledger sequence is "
        "available. The Orchestrator's own ledger judgments (facts, "
        "'is the request satisfied', 'is progress being made', 'is in loop', "
        "next speaker) appear VERBATIM inside its own messages in Section 5 - "
        "read them there and quote them with their msg index.",
        "", "=" * 70,
        "SECTION 4 - TOOL CALL SEQUENCE",
        "=" * 70,
        "NOTE: structured tool-call log was not flushed. Tool invocations and "
        "their results are narrated inside the specialist messages in Section 5 "
        "(e.g. ComputerTerminal command output, WebSurfer page descriptions); "
        "use those narrations as tool evidence.",
        "", "=" * 70,
        f"SECTION 5 - MESSAGE TRANSCRIPT RECONSTRUCTED FROM framework.log "
        f"({dmeta['n_messages']} messages; {dmeta['n_fully_kept']} shown in "
        f"full, {dmeta['n_sampled_out']} sampled out as gists; indices are "
        "RECONSTRUCTED positions; use them in evidence)",
        "=" * 70,
        dialogue,
    ]
    meta = {
        "trace_source": "framework_log_reconstructed",
        "n_messages": dmeta["n_messages"],
        "n_tool_calls": None,
        "n_ledger_entries": None,
        "dialogue_meta": dmeta,
        "pack_chars": sum(len(s) for s in sections),
    }
    return {"pack_text": "\n".join(sections), "case_info": info, "meta": meta}


def build_evidence_pack(logs_dir: Path, request_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    model_path = resolve_model_path(request, request_path)

    messages = json.loads((logs_dir / "messages.json").read_text(encoding="utf-8"))
    orch = json.loads((logs_dir / "orchestration.json").read_text(encoding="utf-8"))
    tc = json.loads((logs_dir / "tool_calls.json").read_text(encoding="utf-8"))

    dialogue, dmeta = build_dialogue(messages)
    info = _extract_case_info(request, ck)

    sections = [
        "=" * 70,
        "SECTION 1 - SYSTEM UNDER TEST (agent roles & coordination rules)",
        "=" * 70,
        build_agent_profile(model_path),
        "", "=" * 70,
        "SECTION 2 - TASK, OUTCOME AND ORACLE",
        "=" * 70,
        format_case_info(info),
        "", "=" * 70,
        "SECTION 3 - COORDINATION TRACE (progress ledger OR selector/runtime trace)",
        "=" * 70,
        format_orchestration(orch),
        "", "=" * 70,
        "SECTION 4 - TOOL CALL SEQUENCE",
        "=" * 70,
        format_tool_calls(tc),
        "", "=" * 70,
        f"SECTION 5 - MESSAGE TRANSCRIPT ({dmeta['n_messages']} messages; "
        f"{dmeta['n_fully_kept']} shown in full, {dmeta['n_sampled_out']} sampled out "
        f"as gists; msg indices are ORIGINAL positions; use them in evidence)",
        "=" * 70,
        dialogue,
    ]
    meta = {
        "n_messages": dmeta["n_messages"],
        "n_tool_calls": len(tc.get("entries", [])),
        "n_ledger_entries": len(orch.get("entries", [])),
        "dialogue_meta": dmeta,
        "pack_chars": sum(len(s) for s in sections),
    }
    return {"pack_text": "\n".join(sections), "case_info": info, "meta": meta}



def build_system_prompt() -> str:
    fm_listing = "\n".join(
        f"{code} {name}\n    Definition: {definition}"
        for code, name, definition in FM_DEFINITIONS
    )
    env_listing = "\n".join(f"- {t}" for t in ENV_BLOCKER_TYPES)
    schema = {
        "task_failed": "boolean: true only if the MAS objectively failed the task "
                       "(wrong answer, no answer, or aborted before completion). A "
                       "correct answer after a messy trajectory is false.",
        "environmental_blockers": [{"type": f"one of: {', '.join(ENV_BLOCKER_TYPES)}",
                                    "evidence": "short quote + msg index"}],
        "fms": {code: {"hit": "boolean",
                        "confidence": "high|medium|low",
                        "reason": "English, 1-3 sentences, why this behavior matches "
                                  "the definition; 'No evidence in the trace.' when false",
                        "evidence": [{"msg_index": "integer original msg index, or null",
                                      "agent": "agent name",
                                      "quote": "short EXACT English quote from the trace"}]}
                 for code in FM_CODES},
        "summary": "one English paragraph: what actually happened and which failure "
                   "modes drove the outcome",
    }
    return f"""You are a senior failure analyst for multi-agent LLM systems. You apply \
MAST (the Multi-Agent System Failure Taxonomy from "Why Do Multi-Agent LLM Systems \
Fail?", arXiv:2503.13657) to ONE execution trace of a multi-agent team solving a task. \
The concrete system (agents, tools, coordination mechanism) is described in Section 1; \
it may be a star-topology orchestrator team or a selector group-chat team. Apply the \
same 14-mode standard regardless of the framework.

THE 14 FAILURE MODES (verbatim definitions):
{fm_listing}

ANNOTATION RULES (from the paper):
1. Multiple modes can be hit on one trace; evaluate all 14 independently.
2. A failure behavior counts EVEN IF the agents later self-corrected it.
3. A mode is hit only when there is concrete behavior in THIS trace. Do not infer \
failure from task difficulty or from a wrong final answer alone.
4. Environmental problems are NOT MAST failure modes: CAPTCHAs, HTTP 403/CSP blocks, \
API billing/quota errors (e.g. HTTP 402), browser/process crashes, network/DNS \
failures, sandbox permission denials, or being killed by the wallclock timeout. Put \
these in "environmental_blockers" instead. The MAS reaction TO an environmental \
problem (repeating a dead path, fabricating an answer, premature giving up) can still \
be an FM - judge the reaction separately.
5. Evidence discipline: if hit=true you MUST provide at least one evidence item with \
the original msg index (as printed in the transcript section) and a SHORT EXACT \
quote. If you cannot quote supporting text, set hit=false.
6. Distinguish near-modes carefully:
   - FM-1.3 step repetition vs FM-2.3 derailment: repetition is re-doing the same \
step; derailment is drifting to a different/irrelevant objective.\
   - FM-1.5 vs FM-2.3 when the task is already done: FM-1.5 is pointlessly \
continuing the SAME task; FM-2.3 is drifting to a DIFFERENT objective that was \
never requested. Concrete FM-2.3 examples: after the user deliverable is produced \
and verified, the agents start debugging/patching/modifying their OWN framework, \
libraries, virtual environments, tool definitions, or test harness; chase an \
unrelated side quest; or try to "fix the tools" instead of working around them. \
Patching files inside site-packages / the agent runtime is FM-2.3 (it may \
co-occur with FM-1.5).
   - FM-1.5 unaware of termination vs FM-3.1 premature termination: FM-1.5 is \
continuing when done (or clearly should stop); FM-3.1 is stopping too early.
   - FM-2.4 withholding (agent HAS info, does not share) vs FM-2.5 ignoring input \
(disregards what another agent said) vs FM-1.4 history loss (context truncation).
   - FM-2.6 reasoning-action mismatch: the agent's own stated reasoning contradicts \
its action (e.g. ledger says request NOT satisfied, then it outputs a fabricated \
answer and terminates).
   - FM-3.2 no/incomplete verification vs FM-3.3 incorrect verification: 3.2 = check \
was skipped or superficial; 3.3 = a check was done but wrongly accepted/rejected.
7. Section 3 contains the coordination trace: a progress ledger (star topology) OR a \
selector/runtime trace with the speaker sequence and injected protocol. Such records \
are strong evidence but are produced by the LLM/manager itself; cross-check them \
against the actual messages (Section 5) and tool calls (Section 4).
8. Be conservative about FM-1.1 and FM-1.2: only mark when a stated task constraint \
or a clearly defined role boundary is violated, not when an agent merely uses an \
unexpected method.
9. Everything you write (reason, summary) MUST be in English. Quotes stay in their \
original language as they appear.
10. No answer key is given. Judge ONLY against the literal task wording. For numeric \
tasks with rounding/unit requirements, do the unit conversion yourself before \
deciding compliance (e.g. 1 Angstrom = 100 picometer, so a value in Angstrom \
"rounded to the nearest picometer" is rounded to TWO decimals; 1.456 A = 145.6 pm \
-> 146 pm -> 1.46 A is COMPLIANT). A compliant answer is never FM-1.1 even if you \
would personally prefer more decimals.
11. FM-3.1 boundary: when the run ends because the round budget was exhausted \
(BUDGET_EXHAUSTED / "Max rounds reached") or the process was killed by a wallclock \
timeout, that termination is FRAMEWORK-IMPOSED, not FM-3.1. Label FM-3.1 only when \
the Orchestrator voluntarily declares completion / emits the final answer while \
budget still remained and objectives were unmet. A fabricated last-second answer at \
budget exhaustion belongs to FM-2.6 and FM-3.2, not FM-3.1.
12. FM-1.2 explicit check: role boundaries are those defined for the SYSTEM UNDER TEST \
in Section 1 (agent roles and runtime tool bindings). Scan for delegations that cross \
them - e.g. an Orchestrator ordering a READ-ONLY navigation/preview agent (FileSurfer) \
to save/create/overwrite files, an agent being asked to do another agent's declared \
job, or a selector team assigning a task to an agent that has neither the role nor any \
bound tool for it. Each such delegation is FM-1.2 even if the specialist refuses, \
fails, or only pretends to comply.
13. FM-2.5 explicit check: scan for concrete results or recommendations returned by \
one agent (e.g. a computed answer in a tool output) that the coordinator or another \
agent then disregards or reverses without evidence-based justification. Overruling a \
correct specialist result is FM-2.5.
14. FM-3.3 explicit check: claiming that an artifact was "verified" / "works" / \
"done" when the verification step was skipped, superficial, or its own evidence \
showed the opposite (e.g. a file still holding old content) is FM-3.3; it may \
co-occur with FM-2.6.
15. Selector group-chat specifics: in teams where a manager picks every speaker \
(Section 3 shows selections, no ledger), the coordination failures still apply - \
e.g. repeatedly re-selecting the same stuck agent is evidence for FM-1.3 ONLY when \
its content redoes the same step without progress (see rule 16); an agent emitting \
the termination token while open questions remain is FM-3.1; failing to request the \
one expert whose role matches a missing step can be FM-2.2; peers ignoring a named \
hand-off request can be FM-2.5. Use agents' own progress statements in Section 5 the \
same way you would use ledger entries.
16. CRITICAL anti-false-positive for FM-1.3 in tool-using teams: ONE agent step can \
appear as several ADJACENT transcript entries - (a) the tool-call arguments, (b) the \
tool result/observation with the same content, and (c) the agent's next message \
narrating or presenting that same result to the team. These adjacent near-duplicates \
are ONE step, NOT repetition. Likewise, presenting a finished result to the user once \
is not repetition. FM-1.3 requires the SAME meaningful action to be issued AGAIN \
after it already produced its result, separated in time, with no new information or \
progress - typically re-calling the same failed query/command, re-attempting the same \
blocked page, or looping between the same subtasks across non-adjacent turns. A single \
self-test demonstration built once and then reported once is NOT FM-1.3.
17. Budget note: the selector teams in this data may use a small fixed turn budget \
(10). Exhausting that budget is FRAMEWORK-IMPOSED termination (same rule as 11): \
mark the fabricated/degenerate final answer as FM-2.6/FM-3.2, not FM-3.1.

Respond with ONE JSON object and nothing else (no markdown fences, no prose). Exact \
shape:
{json.dumps(schema, indent=2, ensure_ascii=False)}"""



def loose_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        candidate = cleaned[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
    raise ValueError("could not parse model output as JSON")


def validate_annotation(obj: Any, outcome: Any = None) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    forced_no_31 = isinstance(outcome, str) and (
        "BUDGET_EXHAUSTED" in outcome or "TIMEOUT" in outcome.upper())
    if not isinstance(obj, dict):
        return None, ["top-level JSON is not an object"]
    fms = obj.get("fms")
    if not isinstance(fms, dict):
        return None, ["missing or malformed 'fms' object"]
    normalized: dict[str, Any] = {}
    for code in FM_CODES:
        item = fms.get(code)
        if not isinstance(item, dict):
            warnings.append(f"{code} missing/malformed -> treated as not hit")
            normalized[code] = {"hit": False, "confidence": "low",
                                "reason": "Annotator did not return this mode.",
                                "evidence": []}
            continue
        hit = bool(item.get("hit", False))
        conf = str(item.get("confidence", "low")).lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        evidence = item.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence = [e for e in evidence if isinstance(e, dict)]
        if hit and not evidence:
            warnings.append(f"{code} hit=true without evidence -> forced to false")
            hit = False
        if code == "FM-3.1" and hit and forced_no_31:
            warnings.append("FM-3.1 hit=true on budget/timeout outcome -> forced to false")
            hit = False
        normalized[code] = {
            "hit": hit,
            "confidence": conf,
            "reason": str(item.get("reason", "")).strip() or "No reason provided.",
            "evidence": evidence,
        }
    if not isinstance(obj.get("task_failed"), bool):
        warnings.append("task_failed missing/non-bool -> null")
        obj["task_failed"] = None
    blockers = obj.get("environmental_blockers") or []
    if not isinstance(blockers, list):
        blockers = []
    obj["fms"] = normalized
    obj["environmental_blockers"] = blockers
    obj["parse_warnings"] = warnings
    return obj, warnings



def call_llm(profile: dict[str, Any], system_prompt: str, user_text: str,
             outcome: Any = None) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=profile["api_key"], base_url=profile["base_url"],
                    timeout=profile["timeout_sec"], max_retries=2)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]

    def _once(json_mode: bool) -> Any:
        kwargs: dict[str, Any] = {"model": profile["model"], "messages": messages,
                                  "temperature": 0,
                                  "max_tokens": 32768}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        return raw, resp.usage

    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    raw_text = usage = None
    for json_mode in (True, False):
        for attempt in range(2):
            try:
                raw_text, usage = _once(json_mode)
                parsed = loose_json_loads(raw_text)
                annotation, warns = validate_annotation(parsed, outcome)
                if annotation is None:
                    raise ValueError("; ".join(warns))
                return {"annotation": annotation, "raw": raw_text, "usage": usage,
                        "json_mode": json_mode, "attempts": attempts + [
                            {"json_mode": json_mode, "attempt": attempt, "ok": True}]}
            except Exception as exc:
                last_error = exc
                attempts.append({"json_mode": json_mode, "attempt": attempt,
                                 "ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]})
                time.sleep(3 * (attempt + 1))
    return {"annotation": None, "raw": raw_text, "usage": None,
            "error": f"{type(last_error).__name__}: {last_error}",
            "json_mode": None, "attempts": attempts}



def annotate_case(logs_dir: Path, request_path: Path, checkpoint_path: Path,
                  out_path: Path, profile: dict[str, Any],
                  case_id: str | None = None) -> dict[str, Any]:
    if (logs_dir / "messages.json").exists():
        pack = build_evidence_pack(logs_dir, request_path, checkpoint_path)
    elif (logs_dir / "framework.log").exists():
        pack = build_evidence_pack_from_framework(logs_dir, request_path, checkpoint_path)
    else:
        raise FileNotFoundError(f"neither messages.json nor framework.log in {logs_dir}")
    cid = case_id or f"{logs_dir.parent.parent.name}/{logs_dir.parent.name}"
    system_prompt = build_system_prompt()
    user_text = pack["pack_text"]
    if pack["meta"].get("trace_source") == "framework_log_reconstructed":
        user_text += ("\n\nREMINDER: This transcript was reconstructed from "
                      "framework.log after a wallclock timeout. Apply the same "
                      "standard; FM-1.5 (unaware of termination) and FM-1.3 "
                      "(repetition) are especially relevant here.")
    outcome = None
    try:
        ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        outcome = (ckpt.get("last_round") or {}).get("outcome")
    except Exception:
        pass
    result = call_llm(profile, system_prompt, user_text, outcome=outcome)

    record = {
        "case_id": cid,
        "prompt_version": PROMPT_VERSION,
        "model": profile["model"],
        "logs_dir": str(logs_dir),
        "checkpoint": str(checkpoint_path),
        "case_info": pack["case_info"],
        "pack_meta": pack["meta"],
        "trace_source": pack["meta"].get("trace_source", "five_piece"),
        "status": "ok" if result["annotation"] else "parse_error",
    }
    if result["annotation"]:
        record.update(result["annotation"])
    if result.get("error"):
        record["error"] = result["error"]
    record["call_attempts"] = result["attempts"]
    if result["usage"] is not None:
        record["token_usage"] = {
            "prompt_tokens": result["usage"].prompt_tokens,
            "completion_tokens": result["usage"].completion_tokens,
            "total_tokens": result["usage"].total_tokens,
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    os.replace(tmp_path, out_path)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, type=Path, help="per-case 'mas' log dir")
    ap.add_argument("--request", required=True, type=Path, help="worker_request.json")
    ap.add_argument("--checkpoint", required=True, type=Path, help="checkpoint C0NN.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--case-id", default=None)
    ap.add_argument("--llm-config", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs/llm_configs.json")
    ap.add_argument("--profile", default="deepseek")
    ap.add_argument("--model", default=None, help="override model name in the profile")
    ap.add_argument("--pack-only", action="store_true",
                    help="build evidence pack and write it to --out without any LLM call")
    args = ap.parse_args()

    load_env_key()
    if args.pack_only:
        if (args.logs / "messages.json").exists():
            pack = build_evidence_pack(args.logs, args.request, args.checkpoint)
        else:
            pack = build_evidence_pack_from_framework(
                args.logs, args.request, args.checkpoint)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(pack["pack_text"], encoding="utf-8")
        print(f"pack written: {args.out} ({pack['meta']['pack_chars']} chars)", file=sys.stderr)
        return 0

    profile = load_llm_profile(args.llm_config, args.profile)
    if args.model:
        profile["model"] = args.model
    t0 = time.time()
    record = annotate_case(args.logs, args.request, args.checkpoint, args.out,
                           profile, args.case_id)
    dt = time.time() - t0
    hits = [c for c in FM_CODES if record.get("fms", {}).get(c, {}).get("hit")]
    print(f"[{record['status']}] {record['case_id']} model={profile['model']} "
          f"{dt:.1f}s hits={hits} -> {args.out}", file=sys.stderr)
    return 0 if record["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
