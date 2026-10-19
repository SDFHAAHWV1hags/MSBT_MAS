#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

SUITE_ROOT = Path(__file__).resolve().parents[1]
MAGENTIC_WORKER = Path(__file__).with_name('magentic_one_worker.py')
CAPTAIN_WORKER = Path(__file__).with_name('captain_worker.py')
_VENDORED_TP = SUITE_ROOT / 'vendor' / 'magentic_runtime' / 'test_pipeline'
_LEGACY_TP = Path('/home/zs/mas/code-0919/test_pipeline')


def _resolve_runtime_dir() -> Path:
    override = os.getenv('CODE0919_TEST_PIPELINE')
    if override:
        return Path(override)
    if (_VENDORED_TP / 'magentic_exec.py').is_file():
        return _VENDORED_TP
    return _LEGACY_TP


def _is_captain(model: Dict[str, Any]) -> bool:
    fam = str(model.get('family') or '')
    fw = str(model.get('framework') or '')
    return fam.startswith('CaptainAgent') or 'Captain' in fam or 'Captain' in fw


def _resolve_model_path(model: Dict[str, Any], context: Dict[str, Any]) -> Path:
    ctx_path = context.get('model_path')
    if ctx_path and Path(ctx_path).is_file():
        return Path(ctx_path)
    target_id = model.get('model_id')
    if target_id:
        for root in (SUITE_ROOT / 'models' / 'selected', SUITE_ROOT / 'models'):
            if not root.is_dir():
                continue
            for cand in sorted(root.glob('*.json')):
                try:
                    data = json.loads(cand.read_text(encoding='utf-8'))
                except Exception:
                    continue
                if data.get('model_id') == target_id:
                    return cand
    return SUITE_ROOT / 'models' / 'selected' / 'Magentic-One.json'


CODE0919_TP = _resolve_runtime_dir()
VENV_PY = Path(os.getenv('CODE0919_VENV_PYTHON', '/home/zs/mas/.venv/bin/python'))
RESULT_MARKER = '@@RESULT@@'

_TRACE_META_KEYS = (
    'num_candidates', 'num_genes', 'num_dropped', 'legal', 'reached_terminal',
    'termination', 'termination_reason', 'stop_reason', 'num_ledger_rounds',
    'num_ledger_replans', 'num_selections', 'num_tool_call_records',
    'num_tool_return_records', 'num_tool_calls', 'final_state',
    'sim_state_matches', 'path_keys', 'delegated_agents', 'selected_agents',
    'dropped',
)


def execute_mas(*, query: str, model: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    captain = _is_captain(model)
    worker = CAPTAIN_WORKER if captain else MAGENTIC_WORKER

    work_dir = Path(context.get('work_dir') or tempfile.mkdtemp(prefix='mas_work_'))
    work_dir.mkdir(parents=True, exist_ok=True)
    idx = context.get('test_index', 'x')
    log_root = os.getenv('MAS_RUN_LOGS_DIR')
    logs_dir = (Path(log_root) / f'test_{int(idx):02d}') if log_root else (work_dir / f'logs_test_{idx}')
    logs_dir.mkdir(parents=True, exist_ok=True)

    request = {
        'query': query,
        'work_dir': str(work_dir.resolve()),
        'logs_dir': str(logs_dir.resolve()),
        'test_pipeline': str(CODE0919_TP),
        'model_path': str(_resolve_model_path(model, context)),
        'max_turns': int(os.getenv('MAS_MAX_TURNS', '29')),
        'sandbox_timeout': int(os.getenv('MAS_SANDBOX_TIMEOUT', '180')),
        'dep': context.get('file_dependency'),
        'family': str(model.get('family') or ''),
    }
    req_file = logs_dir / 'worker_request.json'
    req_file.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')

    env = dict(os.environ)
    env.setdefault('MAS_ENV_FILE', '/home/zs/mas/.env')
    task_timeout = int(os.getenv('MAS_TASK_TIMEOUT', '1800'))
    crash_retries = int(os.getenv('MAS_WORKER_CRASH_RETRIES', '4'))
    retry_backoffs = [
        int(x) for x in os.getenv(
            'MAS_WORKER_CRASH_BACKOFFS', '30,60,120,240'
        ).split(',') if x.strip()
    ]
    attempts: List[Dict[str, Any]] = []
    proc = None
    for attempt in range(crash_retries + 1):
        if attempt > 0:
            time.sleep(retry_backoffs[min(attempt - 1, len(retry_backoffs) - 1)])
        t_attempt = time.monotonic()
        try:
            proc = subprocess.run(
                [str(VENV_PY), str(worker), str(req_file)],
                capture_output=True, text=True, timeout=task_timeout, env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                'raw_log': [{'__worker_timeout__': True, 'task_timeout_sec': task_timeout,
                             'logs_dir': str(logs_dir), 'work_dir': str(work_dir),
                             'attempts': attempts}],
                'final_answer': '', 'outcome': 'ERROR:TaskTimeout',
                'failure_signature': None,
                'timings': {'execution_sec': float(task_timeout),
                            'worker_stdout_tail': (exc.stdout or '')[-2000:] if isinstance(exc.stdout, str) else ''},
            }
        attempts.append({
            'attempt': attempt + 1,
            'returncode': proc.returncode,
            'wall_sec': round(time.monotonic() - t_attempt, 2),
        })
        stdout_attempt = proc.stdout or ''
        has_marker = any(ln.startswith(RESULT_MARKER) for ln in stdout_attempt.splitlines())
        if has_marker or proc.returncode >= 0 or attempt == crash_retries:
            break
    (logs_dir / 'worker_attempts.jsonl').write_text(
        '\n'.join(json.dumps(a, ensure_ascii=False) for a in attempts) + '\n',
        encoding='utf-8')

    stdout = proc.stdout or ''
    stderr = proc.stderr or ''
    (logs_dir / 'worker_stderr.log').write_text(stderr, encoding='utf-8')
    line = next((ln[len(RESULT_MARKER):] for ln in reversed(stdout.splitlines())
                 if ln.startswith(RESULT_MARKER)), None)
    if line is None:
        return {
            'raw_log': [{'__worker_error__': 'no result marker', 'returncode': proc.returncode,
                         'logs_dir': str(logs_dir), 'stderr_tail': stderr[-3000:],
                         'attempts': attempts}],
            'final_answer': '', 'outcome': f'ERROR:WorkerCrash(rc={proc.returncode})',
            'failure_signature': None,
            'timings': {'worker_returncode': proc.returncode,
                        'worker_attempts': attempts},
        }
    d = json.loads(line)
    trace = d.get('trace') or {}
    llm_usage = d.get('llm_usage')
    raw_log: List[Dict[str, Any]] = [{
        '__actual_path__': trace.get('genes', []),
        '__trace_meta__': {k: trace.get(k) for k in _TRACE_META_KEYS},
        '__logs_dir__': str(logs_dir),
        '__llm_usage__': llm_usage,
    }]
    timings: Dict[str, Any] = {
        'execution_sec': d.get('duration_s'),
        'num_messages': d.get('num_messages'),
    }
    if isinstance(llm_usage, dict):
        timings.update({
            'llm_num_calls': llm_usage.get('num_calls'),
            'llm_total_llm_wall_ms': llm_usage.get('total_llm_wall_ms'),
            'llm_prompt_tokens': llm_usage.get('prompt_tokens'),
            'llm_completion_tokens': llm_usage.get('completion_tokens'),
            'llm_total_tokens': llm_usage.get('total_tokens'),
        })
    return {
        'raw_log': raw_log,
        'final_answer': d.get('final_answer') or '',
        'outcome': d.get('outcome', 'UNKNOWN'),
        'failure_signature': None,
        'timings': timings,
    }


def analyze_execution_log(*, raw_log: List[Dict[str, Any]], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    for rec in raw_log or []:
        genes = rec.get('__actual_path__') if isinstance(rec, dict) else None
        if genes is not None:
            return [{'rule_id': g.get('rule_id', ''), 'event': g.get('event', ''),
                     'params': g.get('params', {}) or {}} for g in genes]
    return []
