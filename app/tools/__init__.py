"""Tools exposed to the agent.

Each module owns one tool and its failure handling. Provider-specific
imports stay inside the tool, so a missing or misconfigured integration
degrades that tool alone instead of breaking tool collection.
"""

from app.tools.rag_tool import RAG
from app.tools.web_search import search_web

__all__ = ["RAG", "search_web"]
