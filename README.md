# Offline Agentic EDA Platform

Local, framework-free NLQ data analyst that now supports multi-agent orchestration (planner → coder → executor → reviewer → reporter) with reflection loops and memory. Powered by Phi-4 Mini Q6 (via `llama-cpp-python`). Supports CSV, Excel, JSON, Parquet, SQLite DBs, and PDF text.

## Setup
- Python 3.10+ recommended.
- Install deps:
  ```bash
  pip install -r requirements.txt
  ```
- Download the Phi-4 Mini Q6 GGUF and point to it:
  ```bash
  export PHI4_MINI_Q6_PATH=~/models/phi-4-mini-q6.gguf
  ```
  Install llama-cpp if missing: `pip install llama-cpp-python`.

## Run (one-shot agent)
```bash
python offline_agent.py "top 5 customers by revenue" sales.csv
```

More queries:
- `python offline_agent.py "show correlation matrix for numeric columns" data.parquet`
- `python offline_agent.py "find columns with missing values" dataset.xlsx`
- `python offline_agent.py "run SQL: select count(*) from df" data.csv`
- `python offline_agent.py "summarize this pdf" report.pdf`

## Agentic multi-agent CLI
Runs planner → coder → executor → reviewer → reporter with reflection and session memory:
```bash
python agent_cli.py "auto explore the dataset" data.csv --auto-eda --rounds 3
```
Outputs plan, feedback, result preview, and report. Session memory defaults to `.agentic_session.jsonl`.

### Plot outputs
- Correlation heatmap: add action `plot_correlation` (saves to `outputs/correlation.png`).
- Distributions: `plot_distributions` (histograms for numeric cols, `outputs/distributions.png`).
- Outliers: `plot_outliers` (boxplots, `outputs/outliers.png`).
Plots are saved locally (matplotlib Agg backend) to stay offline.

## How it works
1) Planner: Phi-4 Mini Q6 produces JSON steps conditioned on schema.  
2) Coder: emits runnable pandas/DuckDB code for transparency.  
3) Executor: runs plan safely (summaries, filters, SQL, stats, correlations, outliers).  
4) Reviewer: validates output (non-empty, sane types).  
5) Reporter: LLM-written insights + suggested next questions.  
6) Reflection: on failure, planner/coder retry (capped by `--rounds`). Session memory logs each round.

Model hooks are defined in `offline_agent.py` (Phi-4 Mini Q6 via `llama-cpp-python`) and reused by the agentic stack. `roberta_embed` is stubbed for future retrieval use.

## Notes
- SQLite: pass `.db`/`.sqlite` files; planner can use `run_sql`.  
- PDF: extracts text via `pypdf`; `summarize_text` currently truncates—plug your summarizer if needed.  
- Hardware: `phi4_llm` uses all available CPU threads by default; adjust `n_threads` in `phi4_llm` if you want to cap cores.  
- Memory: agentic CLI writes `.agentic_session.jsonl` (session) and `.agentic_knowledge.jsonl` (placeholder) for long-term storage; both are local/offline.
