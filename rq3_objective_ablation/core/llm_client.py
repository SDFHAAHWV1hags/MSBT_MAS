#!/usr/bin/env python3
from __future__ import annotations
import abc, json, os, urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChatLLM(abc.ABC):
    @abc.abstractmethod
    def complete(self, messages: List[Dict[str,str]], *, temperature: float=0.2,
                 max_tokens: int=16000, response_format_json: bool=False) -> str:
        raise NotImplementedError


class OpenAICompatibleChatLLM(ChatLLM):
    def __init__(self, *, base_url: str, model: str, api_key_env: str, timeout_sec: int=180,
                 extra_headers: Optional[Dict[str,str]]=None):
        self.base_url=base_url.rstrip('/')
        self.model=model
        self.api_key_env=api_key_env
        self.timeout_sec=timeout_sec
        self.extra_headers=extra_headers or {}

    def complete(self, messages, *, temperature=0.2, max_tokens=16000, response_format_json=False):
        key=os.getenv(self.api_key_env)
        if not key:
            raise RuntimeError(f'Missing API key environment variable: {self.api_key_env}')
        payload={'model':self.model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}
        if response_format_json:
            payload['response_format']={'type':'json_object'}
        req=urllib.request.Request(
            self.base_url+'/chat/completions',
            data=json.dumps(payload).encode('utf-8'),method='POST',
            headers={'Content-Type':'application/json','Authorization':f'Bearer {key}',**self.extra_headers},
        )
        with urllib.request.urlopen(req,timeout=self.timeout_sec) as resp:
            data=json.loads(resp.read().decode('utf-8'))
        return str(data['choices'][0]['message']['content'])


class MockChatLLM(ChatLLM):
    def complete(self, messages, *, temperature=0.2, max_tokens=16000, response_format_json=False):
        if response_format_json:
            return json.dumps({'query':'Use a public online source, inspect the internal content of the relevant document, extract and analyze the needed information, perform any required executable calculation, and independently verify the result before returning the requested answer.', 'candidates':['Use a public online source, inspect the internal content of the relevant document, extract and analyze the needed information, perform any required executable calculation, and independently verify the result before returning the requested answer.']})
        return 'Use a public online source, inspect the internal content of the relevant document, extract and analyze the needed information, perform any required executable calculation, and independently verify the result before returning the requested answer.'


def load_llm(config_path: Path, llm_id: str, *, allow_mock: bool=False) -> ChatLLM:
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if llm_id not in cfg['llms']:
        raise KeyError(f'Unknown llm_id={llm_id}; available={list(cfg["llms"])}')
    c=cfg['llms'][llm_id]
    if c.get('type')=='mock':
        if not allow_mock: raise RuntimeError('Mock LLM disabled for experiment runs.')
        return MockChatLLM()
    if c.get('type')!='openai_compatible':
        raise ValueError('Supported types: openai_compatible, mock')
    return OpenAICompatibleChatLLM(
        base_url=c['base_url'],model=c['model'],api_key_env=c['api_key_env'],
        timeout_sec=int(c.get('timeout_sec',180)),extra_headers=c.get('extra_headers',{}),
    )
