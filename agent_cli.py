"""
CLI entrypoint for the agentic offline EDA platform.

Examples:
  python agent_cli.py "auto explore the dataset" data.csv --auto-eda
  python agent_cli.py "top 5 customers by revenue" sales.parquet
"""

from agentic.orchestrator import run_cli

if __name__ == "__main__":
    run_cli()
