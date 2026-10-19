#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
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
    import captain_exec
    import captain_trace
    from adaptive_nsga2 import ModelRuntime

    work_dir = Path(req['work_dir']); work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(req['logs_dir']); logs_dir.mkdir(parents=True, exist_ok=True)

    suite_model = json.loads(Path(req['model_path']).read_text(encoding='utf-8'))
    budget = int((suite_model.get('configuration_parameters') or {}).get('max_interactions') or 10)

    cfg = cfg_mod.load_config(None)
    cfg['_sandbox_mode'] = 'local'
    cfg['_sandbox_timeout'] = int(req.get('sandbox_timeout', 180))
    cfg['_sandbox_image'] = 'mas-sandbox:latest'
    cfg['_mas_max_turns'] = budget

    try:
        rec = asyncio.run(captain_exec.run_captain(
            req['query'], cfg, suite_model,
            sandbox_dir=work_dir,
            logs_dir=logs_dir,
            prefix='mas',
            logger=None,
        ))
        runtime = ModelRuntime(suite_model)
        trace = captain_trace.trace_to_path(
            rec.get('messages') or [], runtime,
            rec.get('outcome', 'UNKNOWN'),
            orchestration=rec.get('orchestration'),
            stop_reason=rec.get('last_stop_reason'),
            tool_calls=rec.get('tool_calls'),
        )
        captain_trace.save_path(trace, logs_dir / 'mas' / 'actual_path.json')
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
