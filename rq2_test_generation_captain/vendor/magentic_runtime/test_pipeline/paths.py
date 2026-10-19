#!/usr/bin/env python3
import os
import sys
from pathlib import Path

TEST_PIPELINE = Path(__file__).resolve().parent
PACKAGE_ROOT = TEST_PIPELINE.parent

V2_DIR = TEST_PIPELINE / "adaptive_nsga2_complete_path_v2"
NLP_DIR = TEST_PIPELINE / "nlp_form_testcase_generation_v1_1"

VENDOR_DIR = TEST_PIPELINE / "vendor"
FUNCTIONS_LIB_SRC = VENDOR_DIR / "functions.py"

DATA_DIR = PACKAGE_ROOT / "data"
GAIA_2023 = DATA_DIR / "GAIA" / "2023"
ASSISTANTBENCH_DIR = DATA_DIR / "AssistantBench"
WHO_WHEN_DIR = DATA_DIR / "Who&When"

MODEL_PATH = V2_DIR / "models" / "running_example_magentic_one_full.json"

CATALOG_PATH = NLP_DIR / "libraries" / "entity_fact_catalog.json"

SANDBOX_DIR = TEST_PIPELINE / "sandbox"
OUT_DIR = TEST_PIPELINE / "output"
RUNS_DIR = OUT_DIR / "runs"

CONFIG_PATH = TEST_PIPELINE / "config.yaml"

def _resolve_env_path() -> Path:
    explicit = os.environ.get("MAS_ENV_FILE")
    if explicit:
        return Path(explicit).expanduser()
    for parent in [PACKAGE_ROOT, *PACKAGE_ROOT.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return PACKAGE_ROOT.parent / ".env"


ENV_PATH = _resolve_env_path()

for _p in (V2_DIR, NLP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

for _d in (SANDBOX_DIR, OUT_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def new_run_dir(when=None) -> Path:
    from datetime import datetime

    d = RUNS_DIR / (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d
