import os

from offline_agent import get_llm as offline_get_llm  # reuse the LLM selector

# Default max rounds for reflection
MAX_REFLECTION_ROUNDS = int(os.getenv("AGENTIC_MAX_ROUNDS", "3"))

# Row cap to prevent OOM; rows above cap are sampled
MAX_ROWS = int(os.getenv("AGENTIC_MAX_ROWS", "200000"))

# Memory file defaults
SESSION_MEMORY_PATH = os.getenv("AGENTIC_SESSION_MEMORY", ".agentic_session.jsonl")
KNOWLEDGE_MEMORY_PATH = os.getenv("AGENTIC_KNOWLEDGE_MEMORY", ".agentic_knowledge.jsonl")
PROFILE_CACHE_PATH = os.getenv("AGENTIC_PROFILE_CACHE", ".agentic_profiles.json")

# Planner context size for prompts (soft hint)
PLANNER_MAX_CONTEXT = 4096


def get_llm():
    """Return the configured LLM callable (LFM2 default, Phi-4 optional)."""
    return offline_get_llm()
