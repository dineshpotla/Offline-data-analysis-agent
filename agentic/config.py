import os

from offline_agent import phi4_llm  # reuse the existing Phi-4 Mini Q6 hook

# Default max rounds for reflection
MAX_REFLECTION_ROUNDS = int(os.getenv("AGENTIC_MAX_ROUNDS", "3"))

# Memory file defaults
SESSION_MEMORY_PATH = os.getenv("AGENTIC_SESSION_MEMORY", ".agentic_session.jsonl")
KNOWLEDGE_MEMORY_PATH = os.getenv("AGENTIC_KNOWLEDGE_MEMORY", ".agentic_knowledge.jsonl")

# Planner context size for prompts (soft hint)
PLANNER_MAX_CONTEXT = 4096


def get_llm():
    """Return the configured LLM callable (Phi-4 Mini Q6 via offline_agent)."""
    return phi4_llm
