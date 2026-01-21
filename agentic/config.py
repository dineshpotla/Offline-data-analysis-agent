import os

from offline_agent import get_llm  # reuse the LLM selector (Phi-4 or LFM2)

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
    """Return the configured LLM callable (Phi-4 Mini Q6 default, LFM2 optional)."""
    return get_llm()
