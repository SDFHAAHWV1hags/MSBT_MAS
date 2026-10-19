#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult, TerminationCondition
from autogen_agentchat.messages import (
    BaseChatMessage,
    SelectSpeakerEvent,
    StopMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool

import llm_provider
import paths
from magentic_exec import (
    _json_safe,
    _msg_to_json,
    _msg_transcript_line,
    _setup_framework_logger,
)

TERMINATE_TOKEN = "TERMINATE"
DEFAULT_TIMEOUT = 180
CODE_FUNC_NAME = "CodeExecutor"
TAVILY_FUNC_NAME = "tavily_web_search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SEC = 30
TAVILY_RESULT_CHAR_CAP = 8000

BUDGET_STOP_MARKERS = ("maximum number of turns", "maximum number of rounds", "max turns reached")
COMPLETED_STOP_MARKER = "final-answer token terminate"
_TOKEN_LINE_RE_TMPL = r"(?im)^\s*{token}\s*$"


class FinalTokenTermination(TerminationCondition):

    def __init__(self, token: str = TERMINATE_TOKEN) -> None:
        self._token = token
        self._pattern = re.compile(_TOKEN_LINE_RE_TMPL.format(token=re.escape(token)))
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages):
        for message in messages:
            if not isinstance(message, BaseChatMessage):
                continue
            if str(getattr(message, "type", "")) not in (
                "TextMessage", "HandoffMessage", "MultiModalMessage"):
                continue
            text = message.to_text() if hasattr(message, "to_text") else ""
            if self._pattern.search(text or ""):
                self._terminated = True
                return StopMessage(
                    content=f"Captain {COMPLETED_STOP_MARKER}d: token {self._token} on its own line",
                    source="FinalTokenTermination",
                )
        return None

    async def reset(self) -> None:
        self._terminated = False

async def tavily_web_search(query: str, max_results: int = 5) -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip().strip("'").strip('"')
    if not api_key:
        return ("Tavily search unavailable: the TAVILY_API_KEY environment variable "
                "is not set. Ask the team to proceed without web search.")
    payload = json.dumps({
        "query": str(query),
        "max_results": max(1, min(int(max_results), 10)),
        "search_depth": "basic",
    }).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    def _do_request() -> Dict[str, Any]:
        with urllib.request.urlopen(request, timeout=TAVILY_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        data = await asyncio.to_thread(_do_request)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            detail = ""
        return f"Tavily search failed: HTTP {exc.code}. {detail}".strip()
    except Exception as exc:
        return f"Tavily search failed: {type(exc).__name__}: {exc}"

    blocks: List[str] = []
    answer = data.get("answer")
    if answer:
        blocks.append(f"ANSWER: {answer}")
    for item in (data.get("results") or [])[:max_results]:
        blocks.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )
    if not blocks:
        return f"No web results found for query: {query}"
    return "\n\n".join(blocks)[:TAVILY_RESULT_CHAR_CAP]


class CaptainModelError(RuntimeError):
    pass


def parse_model(model: Dict[str, Any]) -> Dict[str, Any]:
    agents = list((model.get("structure") or {}).get("agents") or [])
    definitions = model.get("agent_definitions") or {}
    missing = [a for a in agents if not (definitions.get(a) or {}).get("system_prompt")]
    if missing:
        raise CaptainModelError(f"agent_definitions missing system prompt: {missing}")
    bindings = [(b["agent"], b["tool"]) for b in
                (model.get("structure") or {}).get("agent_tool_bindings", [])]
    unknown_agents = {a for a, _ in bindings if a not in agents}
    if unknown_agents:
        raise CaptainModelError(f"bindings reference experts not in structure.agents: {sorted(unknown_agents)}")
    budget = int((model.get("configuration_parameters") or {}).get("max_interactions") or 0)
    if budget <= 0:
        raise CaptainModelError("configuration_parameters.max_interactions missing or non-positive")
    return {"agents": agents, "definitions": definitions, "bindings": bindings, "budget": budget}


def _agent_description(name: str, prompt: str) -> str:
    lines = [ln.strip() for ln in prompt.splitlines()]
    for i, ln in enumerate(lines):
        if ln.lower().replace(" ", "") == "##yourrole":
            for cand in lines[i + 1:]:
                if cand:
                    return cand[:300]
    return f"{name}: a specialist expert in this team."


_RUNTIME_PROTOCOL_HEADER = (
    "\n\n## Runtime protocol (harness-injected; not part of your role card)\n"
    "- You work in a selector-driven team of experts. After every message a manager picks "
    "the next speaker from ALL experts (including yourself again). Do the part belonging to "
    "your role; if another expert's contribution is needed, name that expert and state exactly "
    "what they should do.\n"
    "- When the user's task is fully solved and YOU are delivering the complete verified final "
    f"answer, end your message with the standalone token {TERMINATE_TOKEN} on its own final line. "
    f"Never emit {TERMINATE_TOKEN} before a verified final answer is ready.\n"
)
_RUNTIME_TOOL_HINTS = {
    "Computer_terminal": (
        "- A code execution tool is bound directly to you: call the `CodeExecutor` function with "
        "one COMPLETE python or shell code block. It executes in the shared working directory; "
        "use print() for output; the tool cannot ask follow-ups or edit your code."
    ),
    "Bing_Search_API": (
        f"- A web search tool is bound directly to you: call `{TAVILY_FUNC_NAME}` with a single "
        "natural-language `query` (and optional `max_results`). This replaces any 'Bing Search "
        "API' mentioned in your role card - do NOT hand-write HTTP/search-API code, just call "
        "the function."
    ),
}


def build_system_message(prompt: str, model_tools: List[str]) -> str:
    suffix = _RUNTIME_PROTOCOL_HEADER
    for t in model_tools:
        hint = _RUNTIME_TOOL_HINTS.get(t)
        if hint:
            suffix += hint + "\n"
    return prompt.rstrip() + suffix


def _build_agent_tools(
    bindings: List[Tuple[str, str]], executor: LocalCommandLineCodeExecutor
) -> Tuple[Dict[str, List[Any]], Dict[str, str]]:
    agent_tools: Dict[str, List[Any]] = {}
    func_map: Dict[str, str] = {}
    for agent, model_tool in bindings:
        if model_tool == "Computer_terminal":
            tool: Any = PythonCodeExecutionTool(executor)
            func_map[CODE_FUNC_NAME] = "Computer_terminal"
        elif model_tool == "Bing_Search_API":
            tool = FunctionTool(
                tavily_web_search,
                name=TAVILY_FUNC_NAME,
                description=(
                    "Search the public web via Tavily. Parameters: query (str, one natural-"
                    "language search query), max_results (int, default 5). Returns plain text "
                    "with titles, URLs and content snippets. Use for any up-to-date or external "
                    "fact and for cross-checking."
                ),
            )
            func_map[TAVILY_FUNC_NAME] = "Bing_Search_API"
        else:
            raise CaptainModelError(
                f"executor does not know how to bind tool entity {model_tool!r} (expert {agent})."
                f"only Computer_terminal / Bing_Search_API(Tavily) are currently supported."
            )
        agent_tools.setdefault(agent, []).append(tool)
    return agent_tools, func_map


def _tool_arguments_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _utc_now() -> str:
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def run_captain(
    task: str,
    cfg: Dict[str, Any],
    model: Dict[str, Any],
    sandbox_dir: Path = paths.SANDBOX_DIR,
    logs_dir: Optional[Path] = None,
    prefix: str = "captain",
    logger=None,
) -> Dict[str, Any]:
    parsed = parse_model(model)
    agents_spec = parsed["agents"]
    definitions = parsed["definitions"]
    bindings = parsed["bindings"]
    budget = parsed["budget"]

    logs_dir = Path(logs_dir or paths.RUNS_DIR) / prefix
    logs_dir.mkdir(parents=True, exist_ok=True)
    framework_log = logs_dir / "framework.log"
    transcript_path = logs_dir / "transcript.txt"
    messages_path = logs_dir / "messages.json"
    orchestration_path = logs_dir / "orchestration.json"
    tool_calls_path = logs_dir / "tool_calls.json"
    llm_calls_path = logs_dir / "llm_calls.json"

    timeout = int(cfg.get("_sandbox_timeout", DEFAULT_TIMEOUT))
    max_turns = int(cfg.get("_mas_max_turns") or budget)
    if max_turns != budget:
        max_turns = budget

    file_handler = _setup_framework_logger(framework_log)

    client, settings = llm_provider.build_agent_client(cfg)
    llm_tracker = getattr(client, "mas_llm_tracker", None)
    llm_usage = None
    executor = LocalCommandLineCodeExecutor(work_dir=Path(sandbox_dir), timeout=timeout)
    await executor.start()

    agent_tools, func_map = _build_agent_tools(bindings, executor)
    participants: List[AssistantAgent] = []
    for name in agents_spec:
        prompt = definitions[name]["system_prompt"]
        bound_tools = [t for a, t in bindings if a == name]
        participants.append(AssistantAgent(
            name=name,
            model_client=client,
            tools=agent_tools.get(name),
            system_message=build_system_message(prompt, bound_tools),
            description=_agent_description(name, prompt),
        ))

    team = SelectorGroupChat(
        participants,
        model_client=client,
        termination_condition=FinalTokenTermination(TERMINATE_TOKEN),
        max_turns=max_turns,
        allow_repeated_speaker=True,
        emit_team_events=True,
    )

    if logger:
        logger.info(f"Captain team: {len(participants)} experts {agents_spec}")
        logger.info(f"tool bindings (model ground truth): {bindings or '(no tools, dialogue only)'}")
        logger.info(f"budget max_turns={max_turns} (= model max_interactions), allow_repeated_speaker=True")

    all_messages: List[Dict[str, Any]] = []
    transcript_lines: List[str] = [
        f"# CaptainAgent solve log  {datetime.now().isoformat(timespec='seconds')}",
        f"# model={model.get('model_id')} provider={settings['provider']} model_name={settings['model']}",
        f"# agents={agents_spec} budget={budget} bindings={bindings}",
        "=" * 70,
    ]
    selections: List[Dict[str, Any]] = []
    tool_sink: List[Dict[str, Any]] = []
    pending_calls: Dict[str, Dict[str, Any]] = {}
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None
    outcome = "UNKNOWN"
    t0 = time.perf_counter()

    try:
        async for message in team.run_stream(task=task):
            if isinstance(message, TaskResult):
                stop_reason = message.stop_reason
                last_chat = None
                for m in reversed(message.messages):
                    mtype = getattr(m, "type", "")
                    if mtype in ("TextMessage", "ToolCallSummaryMessage", "MultiModalMessage"):
                        last_chat = m
                        break
                final_answer = str(getattr(last_chat, "content", "")) if last_chat else None
                transcript_lines += [
                    "\n" + "=" * 70,
                    f"[TaskResult] stop_reason={stop_reason}",
                    f"final content: {final_answer}",
                ]
                continue

            rec = _msg_to_json(message)
            all_messages.append(rec)
            transcript_lines.append(_msg_transcript_line(message))

            if isinstance(message, SelectSpeakerEvent):
                chosen = (message.content or [None])[0]
                selections.append({"order": len(selections) + 1, "speaker": chosen})
            elif isinstance(message, ToolCallRequestEvent):
                for fc in message.content:
                    rec_entry = {
                        "seq": len(tool_sink) + 1,
                        "ts": _utc_now(),
                        "agent": str(message.source),
                        "function": str(fc.name),
                        "tool": func_map.get(str(fc.name)),
                        "call_id": str(fc.id),
                        "arguments": _tool_arguments_json(fc.arguments),
                        "is_error": None,
                        "origin": "team_event",
                    }
                    tool_sink.append(rec_entry)
                    pending_calls[str(fc.id)] = rec_entry
            elif isinstance(message, ToolCallExecutionEvent):
                for result in message.content:
                    cid = str(getattr(result, "call_id", ""))
                    req = pending_calls.pop(cid, None)
                    tool_sink.append({
                        "seq": len(tool_sink) + 1,
                        "ts": _utc_now(),
                        "agent": str(message.source),
                        "function": str(getattr(result, "name", "") or (req or {}).get("function", "")),
                        "tool": (req or {}).get("tool") or func_map.get(str(getattr(result, "name", ""))),
                        "call_id": cid,
                        "is_error": bool(getattr(result, "is_error", False)),
                        "result_excerpt": str(getattr(result, "content", ""))[:1000],
                        "origin": "team_event",
                        "phase": "return",
                    })

        duration = round(time.perf_counter() - t0, 3)

        last_agent_text = ""
        for m in reversed(all_messages):
            if (m.get("source") in agents_spec
                    and m.get("type") in ("TextMessage", "ToolCallSummaryMessage")):
                c = m.get("content")
                last_agent_text = c if isinstance(c, str) else json.dumps(_json_safe(c), ensure_ascii=False)
                break
        terminated_by_text = bool(re.search(rf"\b{TERMINATE_TOKEN}\b", last_agent_text))
        low_reason = str(stop_reason or "").lower()
        if COMPLETED_STOP_MARKER in low_reason:
            outcome = "COMPLETED"
        elif any(k in low_reason for k in BUDGET_STOP_MARKERS):
            outcome = "BUDGET_EXHAUSTED"
        elif terminated_by_text:
            outcome = "COMPLETED"
        elif len(selections) >= max_turns:
            outcome = "BUDGET_EXHAUSTED"
        else:
            outcome = "UNKNOWN"

        transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
        messages_path.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2,
                                            default=_json_safe), encoding="utf-8")
        orchestration_path.write_text(json.dumps({
            "family": str(model.get("family")),
            "model_id": model.get("model_id"),
            "budget": budget,
            "max_turns": max_turns,
            "bindings": [{"agent": a, "tool": t} for a, t in bindings],
            "function_to_model_tool": func_map,
            "runtime_protocol_header": _RUNTIME_PROTOCOL_HEADER,
            "runtime_tool_hints": _RUNTIME_TOOL_HINTS,
            "num_selections": len(selections),
            "selections": selections,
            "stop_reason": stop_reason,
            "outcome": outcome,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tool_calls_path.write_text(json.dumps({"entries": tool_sink}, ensure_ascii=False,
                                              indent=2, default=_json_safe), encoding="utf-8")
        if llm_tracker is not None:
            llm_usage = llm_tracker.summary()
            llm_calls_path.write_text(
                json.dumps(llm_usage, ensure_ascii=False, indent=2), encoding="utf-8")
        if logger:
            logger.info(f"Captain solve done: selections={len(selections)}/{max_turns} "
                        f"tool events={len(tool_sink)} time={duration}s "
                        f"stop_reason={stop_reason!r} outcome={outcome}")

        return {
            "final_answer": final_answer,
            "outcome": outcome,
            "num_messages": len(all_messages),
            "duration_s": duration,
            "last_stop_reason": stop_reason,
            "max_turns": max_turns,
            "orchestration": {
                "budget": budget,
                "max_turns": max_turns,
                "selections": selections,
                "function_to_model_tool": func_map,
                "stop_reason": stop_reason,
                "outcome": outcome,
            },
            "tool_calls": tool_sink,
            "messages": all_messages,
            "llm_usage": llm_usage,
            "log_paths": {
                "framework": str(framework_log),
                "transcript": str(transcript_path),
                "messages": str(messages_path),
                "orchestration": str(orchestration_path),
                "tool_calls": str(tool_calls_path),
                "llm_calls": str(llm_calls_path),
            },
        }
    except Exception as exc:
        duration = round(time.perf_counter() - t0, 3)
        err = f"ERROR:{type(exc).__name__}: {exc}"
        transcript_lines += ["\n" + "=" * 70, f"[ERROR] {err}", traceback.format_exc()[-4000:]]
        transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
        messages_path.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2,
                                            default=_json_safe), encoding="utf-8")
        orchestration_path.write_text(json.dumps({
            "budget": budget, "max_turns": max_turns, "selections": selections,
            "function_to_model_tool": func_map, "stop_reason": stop_reason,
            "outcome": err,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tool_calls_path.write_text(json.dumps({"entries": tool_sink}, ensure_ascii=False,
                                              indent=2, default=_json_safe), encoding="utf-8")
        if llm_tracker is not None:
            llm_usage = llm_tracker.summary()
            llm_calls_path.write_text(
                json.dumps(llm_usage, ensure_ascii=False, indent=2), encoding="utf-8")
        if logger:
            logger.warn(f"Captain execution error: {err}")
        return {
            "final_answer": final_answer,
            "outcome": err,
            "num_messages": len(all_messages),
            "duration_s": duration,
            "last_stop_reason": stop_reason,
            "max_turns": max_turns,
            "orchestration": {
                "budget": budget, "max_turns": max_turns, "selections": selections,
                "function_to_model_tool": func_map, "stop_reason": stop_reason,
                "outcome": err,
            },
            "tool_calls": tool_sink,
            "messages": all_messages,
            "llm_usage": llm_usage,
            "log_paths": {
                "framework": str(framework_log),
                "transcript": str(transcript_path),
                "messages": str(messages_path),
                "orchestration": str(orchestration_path),
                "tool_calls": str(tool_calls_path),
                "llm_calls": str(llm_calls_path),
            },
            "error": err,
        }
    finally:
        try:
            await executor.stop()
        except Exception:
            pass
        try:
            logging_root = __import__("logging").getLogger()
            logging_root.removeHandler(file_handler)
            file_handler.close()
        except Exception:
            pass
