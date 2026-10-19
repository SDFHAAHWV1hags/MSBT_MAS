#!/usr/bin/env python3
from __future__ import annotations
import abc, importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from adaptive_search import Gene

@dataclass
class ExecutionResult:
    raw_log: List[Dict[str,Any]]
    final_answer: str
    outcome: str
    failure_signature: Optional[str]=None
    timings: Optional[Dict[str,float]]=None

class ExperimentExecutor(abc.ABC):
    @abc.abstractmethod
    def execute(self, query: str, model: Dict[str,Any], context: Dict[str,Any]) -> ExecutionResult: ...
    @abc.abstractmethod
    def analyze_path(self, raw_log: List[Dict[str,Any]], model: Dict[str,Any]) -> List[Gene]: ...

class ModuleExecutor(ExperimentExecutor):
    def __init__(self,module_path: Path):
        spec=importlib.util.spec_from_file_location('mas_harness_plugin',str(module_path))
        if spec is None or spec.loader is None: raise ImportError(module_path)
        self.mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.mod)
    def execute(self,query,model,context):
        d=self.mod.execute_mas(query=query,model=model,context=context)
        return ExecutionResult(
            raw_log=list(d.get('raw_log',[])),final_answer=str(d.get('final_answer','')),
            outcome=str(d.get('outcome','UNKNOWN')),failure_signature=d.get('failure_signature'),
            timings=d.get('timings'),
        )
    def analyze_path(self,raw_log,model):
        xs=self.mod.analyze_execution_log(raw_log=raw_log,model=model)
        out=[]
        for x in xs:
            if isinstance(x,Gene): out.append(x)
            else:
                params=x.get('params',{}) or {}
                out.append(Gene(str(x.get('rule_id','')),str(x.get('event','')),tuple(sorted((str(k),str(v)) for k,v in params.items()))))
        return out

def extract_llm_usage(raw_log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if raw_log and isinstance(raw_log[0], dict):
        u = raw_log[0].get("__llm_usage__")
        return u if isinstance(u, dict) else None
    return None


class MockExecutor(ExperimentExecutor):
    def execute(self,query,model,context):
        target=context.get('target_path',[])
        log=[{'rule_id':g.rule_id,'event':g.event,'params':dict(g.params)} for g in target]
        return ExecutionResult(log,'MOCK_SUCCESS','PASS',None,{'execution_sec':0.0})
    def analyze_path(self,raw_log,model):
        return [Gene(str(x.get('rule_id','')),str(x.get('event','')),tuple(sorted((str(k),str(v)) for k,v in (x.get('params',{}) or {}).items()))) for x in raw_log]
