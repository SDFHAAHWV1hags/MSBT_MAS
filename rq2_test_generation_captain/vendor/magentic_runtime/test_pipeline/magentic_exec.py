#!/usr/bin/env python3
from __future__ import annotations

import ast
import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

import llm_provider
import paths

DOCKER_SANDBOX_ROOT = "/workspace"
DEFAULT_SANDBOX_MODE = "docker"
DEFAULT_TIMEOUT = 180
DEFAULT_IMAGE = "mas-sandbox:latest"
UPSTREAM_IMAGE = "python:3-slim"
DEFAULT_MAX_TURNS = 20


def sandbox_root(mode: str, sandbox_dir: Path) -> str:
    if str(mode).lower() == "local":
        return str(Path(sandbox_dir).resolve())
    return DOCKER_SANDBOX_ROOT


TRACE_LOGGER_NAME = "autogen_agentchat"
_PROGRESS_LEDGER_PREFIX = "Progress Ledger: "
_STALL_REPLAN_MARKER = "Stall count exceeded, re-planning with the outer loop..."
_TASK_COMPLETE_MARKER = "Task completed, preparing final answer..."


class OrchestrationCapture(logging.Handler):

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.entries: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if msg.startswith(_PROGRESS_LEDGER_PREFIX):
            raw = msg[len(_PROGRESS_LEDGER_PREFIX):]
            try:
                ledger = ast.literal_eval(raw)
            except Exception:
                return
            if isinstance(ledger, dict):
                self.entries.append({"kind": "progress_ledger", "ledger": ledger})
        elif msg.startswith(_STALL_REPLAN_MARKER):
            self.entries.append({"kind": "stall_replan"})
        elif msg.startswith(_TASK_COMPLETE_MARKER):
            self.entries.append({"kind": "task_completed"})


ROUND_CAP_MARKERS = ("max rounds", "max turns", "round limit", "turn limit")


def _is_round_cap_reason(stop_reason: Optional[str]) -> bool:
    if not stop_reason:
        return False
    low = str(stop_reason).lower()
    return any(k in low for k in ROUND_CAP_MARKERS)


def _count_ledgers(entries: List[Dict[str, Any]]) -> int:
    return sum(1 for e in entries if e.get("kind") == "progress_ledger")




def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ToolCallSink:

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def record(self, agent: str, action: str, arguments: Any = None, *,
               start_ts: Optional[str] = None, end_ts: Optional[str] = None,
               duration_ms: Optional[float] = None, is_error: bool = False,
               **extra: Any) -> None:
        start_ts = start_ts or _utc_now()
        self.entries.append({
            "seq": len(self.entries) + 1,
            "ts": start_ts,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_ms": duration_ms,
            "is_error": bool(is_error),
            "agent": agent,
            "action": action,
            "arguments": arguments,
            **extra,
        })


def _tool_arguments(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _wrap_web_surfer_tools(agent: Any, sink: ToolCallSink) -> None:
    original = agent._execute_tool
    agent_name = str(getattr(agent, "name", "WebSurfer"))

    async def wrapper(message: Any, rects: Any, tool_names: str,
                      cancellation_token: Any = None) -> Any:
        start_ts, t0 = _utc_now(), time.perf_counter()
        name = "unknown"
        arguments: Any = None
        is_error = False
        try:
            try:
                first = message[0]
                name = str(getattr(first, "name", name))
                arguments = _tool_arguments(getattr(first, "arguments", None))
            except Exception:
                pass
            return await original(message, rects, tool_names,
                                  cancellation_token=cancellation_token)
        except Exception:
            is_error = True
            raise
        finally:
            end_ts, t1 = _utc_now(), time.perf_counter()
            url = None
            try:
                url = agent._page.url
            except Exception:
                pass
            sink.record(agent_name, name, arguments,
                        start_ts=start_ts, end_ts=end_ts,
                        duration_ms=round((t1 - t0) * 1000, 3),
                        is_error=is_error, origin="websurfer_execute", url=url)

    agent._execute_tool = wrapper


_FILE_BROWSER_TOOLS = (
    ("open_path", "open_path", ("path",)),
    ("page_up", "page_up", ()),
    ("page_down", "page_down", ()),
    ("find_on_page", "find_on_page_ctrl_f", ("search_string",)),
    ("find_next", "find_next", ()),
)


def _wrap_file_surfer_tools(agent: Any, sink: ToolCallSink) -> None:
    browser = agent._browser
    agent_name = str(getattr(agent, "name", "FileSurfer"))

    def make_wrapper(method_name: str, tool_name: str, arg_names: tuple):
        original = getattr(browser, method_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_ts, t0 = _utc_now(), time.perf_counter()
            is_error = False
            try:
                return original(*args, **kwargs)
            except Exception:
                is_error = True
                raise
            finally:
                end_ts, t1 = _utc_now(), time.perf_counter()
                named = dict(zip(arg_names, args))
                named.update(kwargs)
                sink.record(agent_name, tool_name, named or None,
                            start_ts=start_ts, end_ts=end_ts,
                            duration_ms=round((t1 - t0) * 1000, 3),
                            is_error=is_error, origin="filesurfer_browser")

        return wrapper

    for method_name, tool_name, arg_names in _FILE_BROWSER_TOOLS:
        setattr(browser, method_name,
                make_wrapper(method_name, tool_name, arg_names))


class _RecordingCodeExecutor:

    def __init__(self, inner: Any, sink: ToolCallSink, agent: str = "ComputerTerminal") -> None:
        self._inner = inner
        self._sink = sink
        self._agent = agent

    async def execute_code_blocks(self, code_blocks: Any, cancellation_token: Any = None) -> Any:
        start_ts, t0 = _utc_now(), time.perf_counter()
        exit_code = None
        is_error = False
        try:
            result = await self._inner.execute_code_blocks(
                code_blocks, cancellation_token=cancellation_token)
            exit_code = getattr(result, "exit_code", None)
            is_error = exit_code not in (0, None)
            return result
        except Exception:
            is_error = True
            raise
        finally:
            end_ts, t1 = _utc_now(), time.perf_counter()
            blocks = list(code_blocks or [])
            self._sink.record(
                self._agent, "execute_code_blocks",
                {"languages": [getattr(b, "language", "") for b in blocks],
                 "num_blocks": len(blocks)},
                start_ts=start_ts, end_ts=end_ts,
                duration_ms=round((t1 - t0) * 1000, 3),
                is_error=is_error, origin="code_executor", exit_code=exit_code)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def create_code_executor(sandbox_dir: Path, timeout: int = DEFAULT_TIMEOUT,
                         image: str = DEFAULT_IMAGE,
                         mode: str = DEFAULT_SANDBOX_MODE):
    sandbox_dir = Path(sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    if str(mode).lower() == "local":
        from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

        return LocalCommandLineCodeExecutor(work_dir=sandbox_dir, timeout=timeout)

    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"sandbox mode requires Docker, but the Docker daemon is unreachable"
            f"({type(exc).__name__}: {exc}). Start Docker first and retry."
        ) from exc

    if image != UPSTREAM_IMAGE:
        try:
            client.images.get(image)
        except Exception as exc:
            raise RuntimeError(
                f"sandbox image '{image}' does not exist locally and is not a directly pullable upstream image.\n"
                f"build it first: docker build -f docker/sandbox.Dockerfile -t {image} .\n"
                f"(or change sandbox.image in config.yaml back to {UPSTREAM_IMAGE})"
            ) from exc

    from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

    return DockerCommandLineCodeExecutor(
        image=image, work_dir=sandbox_dir, timeout=timeout, auto_remove=True)


_TEXT_EXTS = (".csv", ".txt", ".md", ".json", ".tsv", ".log")


def pick_probe_file(seed_map: Dict[str, Dict[str, str]]) -> str:
    names = sorted({info["file_name"] for info in seed_map.values()})
    if not names:
        raise ValueError("no seed dependency files inside the sandbox; cannot perform sandbox self-check")
    for ext in _TEXT_EXTS:
        for n in names:
            if n.lower().endswith(ext):
                return n
    return names[0]


def build_sandbox_probe_task(file_name: str, k: int = 5,
                             root: str = DOCKER_SANDBOX_ROOT) -> str:
    return f"""Run a sandbox self-check and report the findings as a short structured report.

Context: you are expected to be running inside a code-execution sandbox whose
working directory is exactly `{root}`. The file `{file_name}`
should already exist in that working directory.

Perform all three checks and show the raw evidence you collected:

1. FILE HEAD
   Read the first {k} lines of `{file_name}` in the working directory.
   Report those lines verbatim. If the file is binary (not human-readable text),
   say so explicitly and report the raw output anyway.

2. SANDBOX DETECTION
   Determine whether the current environment is a container / sandbox rather than the
   host machine. Collect at least these pieces of evidence:
     - the hostname;
     - the kernel / OS description;
     - whether `/.dockerenv` exists;
     - the contents of `/proc/1/cgroup`.
   Then state your verdict (sandbox: yes / no) and cite which evidence supports it.

3. DIRECTORY CHECK
   List the working directory, print its absolute path, and confirm whether that path is
   exactly `{root}` and whether `{file_name}` is present in it.
   If the file or the directory is NOT as expected, say so explicitly - do not guess.

End your answer with a section titled "FINAL REPORT" containing exactly these lines
(fill in the values you actually observed):

FILE: {file_name}
FIRST_LINES: <the first {k} lines, joined by " | ">
IS_SANDBOX: <YES or NO>
WORKDIR: <absolute path of the working directory>
DIR_AS_EXPECTED: <YES or NO>
FILE_PRESENT: <YES or NO>
"""


def _setup_framework_logger(log_path: Path) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.DEBUG)
    return handler


def _msg_to_json(m: Any) -> Dict[str, Any]:
    if hasattr(m, "model_dump"):
        try:
            return m.model_dump(mode="json")
        except Exception:
            try:
                return m.model_dump()
            except Exception:
                pass
    return {"__repr__": str(m)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    data = getattr(value, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return f"<{type(value).__name__} {len(data)} bytes>"
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    return f"<{type(value).__name__}>"


def _msg_transcript_line(m: Any) -> str:
    typ = type(m).__name__
    src = getattr(m, "source", "")
    content = getattr(m, "content", None)
    if content is None and hasattr(m, "model_dump"):
        try:
            content = json.dumps(_json_safe(m.model_dump(mode="json")), ensure_ascii=False)
        except Exception:
            content = str(m)
    if isinstance(content, (list, dict)):
        content = json.dumps(_json_safe(content), ensure_ascii=False)
    return f"[{typ}] {src}: {content}"


async def run_magentic_one(
    task: str,
    cfg: Dict[str, Any],
    sandbox_dir: Path = paths.SANDBOX_DIR,
    logs_dir: Optional[Path] = None,
    prefix: str = "magentic_one",
    logger=None,
) -> Dict[str, Any]:
    logs_dir = Path(logs_dir or paths.RUNS_DIR) / prefix
    logs_dir.mkdir(parents=True, exist_ok=True)
    framework_log = logs_dir / "framework.log"
    transcript_path = logs_dir / "transcript.txt"
    messages_path = logs_dir / "messages.json"
    orchestration_path = logs_dir / "orchestration.json"
    tool_calls_path = logs_dir / "tool_calls.json"
    llm_calls_path = logs_dir / "llm_calls.json"

    _file_handler = _setup_framework_logger(framework_log)
    capture = OrchestrationCapture()
    logging.getLogger(TRACE_LOGGER_NAME).addHandler(capture)
    sink = ToolCallSink()

    timeout = int(cfg.get("_sandbox_timeout", DEFAULT_TIMEOUT))
    image = str(cfg.get("_sandbox_image") or DEFAULT_IMAGE)
    mode = str(cfg.get("_sandbox_mode") or DEFAULT_SANDBOX_MODE)
    max_turns = int(cfg.get("_mas_max_turns") or DEFAULT_MAX_TURNS)
    root = sandbox_root(mode, sandbox_dir)
    client, settings = llm_provider.build_agent_client(cfg)
    llm_tracker = getattr(client, "mas_llm_tracker", None)
    llm_usage: Optional[Dict[str, Any]] = None
    executor = create_code_executor(sandbox_dir, timeout=timeout, image=image, mode=mode)
    await executor.start()
    file_surfer = FileSurfer("FileSurfer", model_client=client, base_path=root)
    web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client)
    _wrap_file_surfer_tools(file_surfer, sink)
    _wrap_web_surfer_tools(web_surfer, sink)
    team = MagenticOneGroupChat(
        [
            file_surfer,
            web_surfer,
            MagenticOneCoderAgent("Coder", model_client=client),
            CodeExecutorAgent("ComputerTerminal", code_executor=_RecordingCodeExecutor(executor, sink)),
        ],
        model_client=client,
        max_turns=max_turns,
        emit_team_events=True,
    )

    if logger:
        logger.info(f"MAS model: provider={settings['provider']} model={settings['model']} "
                    f"temperature={settings['temperature']}")
        if str(mode).lower() == "local":
            logger.warn(f"sandbox: local  {root} (executed directly on host, NO isolation)   timeout={timeout}s")
        else:
            logger.info(f"sandbox: docker  {sandbox_dir} -> {root}   "
                        f"image={image}   timeout={timeout}s")
        logger.info(f"team limit: max_turns={max_turns} (budget counts orchestrator turns; max_turns should equal budget-1),"
                    f"emit_team_events=True")
        logger.info("tool-call capture (execution-point wrapping + wall-clock timing): WebSurfer=_execute_tool, "
                    "FileSurfer=_browser five methods, ComputerTerminal=execute_code_blocks")

    all_messages: List[Dict[str, Any]] = []
    transcript_lines: List[str] = [
        f"# Magentic-One solve log  {datetime.now().isoformat(timespec='seconds')}",
        f"# provider={settings['provider']} model={settings['model']} sandbox=docker({image})",
        "=" * 70,
    ]
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None
    outcome = "UNKNOWN"
    t0 = time.perf_counter()

    try:
        async for message in team.run_stream(task=task):
            if isinstance(message, TaskResult):
                stop_reason = message.stop_reason
                last = message.messages[-1] if message.messages else None
                final_answer = str(getattr(last, "content", None)) if last else None
                transcript_lines += [
                    "\n" + "=" * 70,
                    f"[TaskResult] stop_reason={stop_reason}",
                    f"final content: {final_answer}",
                ]
                outcome = "COMPLETED" if not _is_round_cap_reason(stop_reason) else "BUDGET_EXHAUSTED"
            else:
                rec = _msg_to_json(message)
                all_messages.append(rec)
                transcript_lines.append(_msg_transcript_line(message))
                if logger:
                    logger.info(f"MAS message #{len(all_messages)-1} "
                                f"[{rec.get('type')}] {rec.get('source')}")

        duration = round(time.perf_counter() - t0, 3)
        transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
        messages_path.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        orchestration_path.write_text(
            json.dumps({"max_turns": max_turns, "stop_reason": stop_reason,
                        "entries": capture.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        tool_calls_path.write_text(
            json.dumps({"entries": sink.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        if llm_tracker is not None:
            llm_usage = llm_tracker.summary()
            llm_calls_path.write_text(
                json.dumps(llm_usage, ensure_ascii=False, indent=2),
                encoding="utf-8")
        if logger:
            logger.info(f"MAS solve done: messages={len(all_messages)}  time={duration}s  "
                        f"stop_reason={stop_reason!r}  turn ledgers={_count_ledgers(capture.entries)}  "
                        f"tool_calls={len(sink.entries)}")
            logger.block("MAS final answer", str(final_answer)[:4000])
            logger.kv({"artifact logs": [framework_log.name, transcript_path.name,
                                  messages_path.name, orchestration_path.name,
                                  tool_calls_path.name]})

        return {
            "final_answer": final_answer,
            "outcome": outcome,
            "sandbox_mode": mode,
            "num_messages": len(all_messages),
            "duration_s": duration,
            "last_stop_reason": stop_reason,
            "max_turns": max_turns,
            "orchestration": capture.entries,
            "tool_calls": sink.entries,
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
        outcome = f"ERROR:{type(exc).__name__}: {exc}"
        transcript_lines += ["\n" + "=" * 70, f"[ERROR] {outcome}"]
        transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
        messages_path.write_text(json.dumps(all_messages, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        orchestration_path.write_text(
            json.dumps({"max_turns": max_turns, "stop_reason": stop_reason,
                        "entries": capture.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        tool_calls_path.write_text(
            json.dumps({"entries": sink.entries}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        if llm_tracker is not None:
            llm_usage = llm_tracker.summary()
            llm_calls_path.write_text(
                json.dumps(llm_usage, ensure_ascii=False, indent=2),
                encoding="utf-8")
        if logger:
            logger.warn(f"MAS execution error: {outcome}")
            logger.block("MAS error stack", traceback.format_exc().strip())
        return {
            "final_answer": final_answer,
            "outcome": outcome,
            "sandbox_mode": mode,
            "num_messages": len(all_messages),
            "duration_s": duration,
            "last_stop_reason": stop_reason,
            "max_turns": max_turns,
            "orchestration": capture.entries,
            "tool_calls": sink.entries,
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
            "error": outcome,
        }
    finally:
        try:
            await executor.stop()
        except Exception:
            pass
        try:
            logging.getLogger(TRACE_LOGGER_NAME).removeHandler(capture)
        except Exception:
            pass
        try:
            if _file_handler is not None:
                logging.getLogger().removeHandler(_file_handler)
                _file_handler.close()
        except Exception:
            pass
