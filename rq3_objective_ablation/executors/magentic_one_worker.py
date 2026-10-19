#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

def _clean_reexec_if_sandboxed() -> None:
    if os.environ.get('_MAS_WORKER_REEXECED') == '1':
        return
    injected = [k for k in os.environ
                if k == 'LD_PRELOAD' or k.startswith('SBOX_')
                or k.startswith('TRAE_SANDBOX_')]
    if not injected:
        return
    env = {k: v for k, v in os.environ.items()
           if k != 'LD_PRELOAD' and not k.startswith('SBOX_')
           and not k.startswith('TRAE_SANDBOX_')}
    env['_MAS_WORKER_REEXECED'] = '1'
    sys.stderr.write(
        f'[worker] stripping trae-sandbox env ({",".join(sorted(injected))}) '
        f'and re-exec clean\n')
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_clean_reexec_if_sandboxed()

try:
    import faulthandler
    faulthandler.enable()
except Exception:
    pass

import asyncio
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    req_path = Path(sys.argv[1])
    req = json.loads(req_path.read_text(encoding='utf-8'))
    tp = Path(req['test_pipeline'])
    sys.path.insert(0, str(tp))
    sys.path.insert(0, str(tp / 'adaptive_nsga2_complete_path_v2'))

    import config as cfg_mod
    import magentic_exec
    import mas_trace
    from adaptive_nsga2 import ModelRuntime

    work_dir = Path(req['work_dir']); work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(req['logs_dir']); logs_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(work_dir)

    cfg = cfg_mod.load_config(os.getenv('MAS_CONFIG_PATH') or None)
    cfg['_sandbox_mode'] = 'local'
    cfg['_sandbox_timeout'] = int(req.get('sandbox_timeout', 180))
    cfg['_sandbox_image'] = 'mas-sandbox:latest'
    cfg['_mas_max_turns'] = int(req.get('max_turns', 29))

    try:
        rec = asyncio.run(magentic_exec.run_magentic_one(
            req['query'], cfg,
            sandbox_dir=work_dir,
            logs_dir=logs_dir,
            prefix='mas',
            logger=None,
        ))
        suite_model = json.loads(Path(req['model_path']).read_text(encoding='utf-8'))
        runtime = ModelRuntime(suite_model)
        trace = mas_trace.trace_to_path(
            rec.get('messages') or [], runtime,
            rec.get('outcome', 'UNKNOWN'),
            orchestration=rec.get('orchestration'),
            stop_reason=rec.get('last_stop_reason'),
            tool_calls=rec.get('tool_calls'),
        )
        mas_trace.save_path(trace, logs_dir / 'mas' / 'actual_path.json')
        result = {
            'final_answer': rec.get('final_answer'),
            'outcome': rec.get('outcome'),
            'duration_s': rec.get('duration_s'),
            'num_messages': rec.get('num_messages'),
            'stop_reason': rec.get('last_stop_reason'),
            'max_turns': rec.get('max_turns'),
            'error': rec.get('error'),
            'trace': trace,
            'llm_usage': rec.get('llm_usage'),
        }
    except Exception as exc:
        result = {
            'final_answer': None,
            'outcome': f'ERROR:{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc()[-4000:],
            'trace': {'genes': []},
            'llm_usage': None,
        }
    sys.stdout.write('\n@@RESULT@@' + json.dumps(result, ensure_ascii=False) + '\n')
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
