# Model- and Search-based Testing for Systematic Behavioral Exploration of LLM-based Multi-Agent System

- **RQ1**: Search performance of multi-objective search algorithm replays (NSGA-II / SPEA-II / objective-guided random) on planning archives
- **RQ2**: Comparison of four problem generation methods (Scheme A / OG-Random / MB-Random / LLM-Direct) across 7 multi-agent systems, evaluated via Failure Mode (FM) annotations
- **RQ3**: Ablation study on the objectives of Scheme A (coverage / diversity / cost)

## Environment Setup

Python 3.14 is required.

```bash
pip install -r requirements.txt
playwright install chromium
export DEEPSEEK_API_KEY=...
```

## Data Download

Input data and full experimental results are released as data packages :

| Data Package                       | Link                                                         | Extracted Path               | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------ |
| `data_inputs_schemea_lanes.tar.gz` | https://github.com/SDFHAAHWV1hags/MSBT_MAS/releases/download/v2/data_inputs_schemea_lanes.tar.gz | `data/inputs/schemea_lanes/` | Input data: planning archives (lane) for each system and seed; inputs for Scheme A problem generation and RQ1 replay |
| `results_rq1.tar.gz`               | https://github.com/SDFHAAHWV1hags/MSBT_MAS/releases/download/v2/results_rq1.tar.gz | `results/rq1/`               | Full RQ1 results: 8400 replayed search archives and metrics  |
| `results_rq2.tar.gz`               | https://github.com/SDFHAAHWV1hags/MSBT_MAS/releases/download/v2/results_rq2.tar.gz | `results/rq2/`               | Full RQ2 results: generated lanes, MAS execution logs, LLM call records, FM annotations |
| `results_rq3.tar.gz`               | https://github.com/SDFHAAHWV1hags/MSBT_MAS/releases/download/v2/results_rq3.tar.gz | `results/rq3/`               | Full RQ3 results: lanes from objective ablation, execution logs, FM annotations |
| `results_analysis.tar.gz`          | https://github.com/SDFHAAHWV1hags/MSBT_MAS/releases/download/v2/results_analysis.tar.gz | `results/analysis/`          | Aggregated CSV files with paper statistics                   |


## Library Description

The BRS rule library (6 JSON files located under `brs_v0_4/library/` in each code tree) supports the path-to-user-request problem generation stage. Given planning paths produced by search, the problem generator first projects them into behavioral signatures. It then grounds the paths using the rule library: matching task frameworks corresponding to behaviors in the task snippet library, selecting source tasks from the grounding instance library as corpus bases, and instantiating paths into concrete user requests under constraints of operation/data type/intent vocabulary and transition semantic mappings.
Scheme A, OG-Random and MB-Random problem generation methods in RQ2, as well as RQ3, rely on this library. RQ1 replay and LLM-Direct in RQ2 do not require it.

This library is not publicly available for now and will be released upon formal publication of the paper.