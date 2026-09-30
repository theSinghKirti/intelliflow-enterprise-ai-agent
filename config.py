"""Configuration — all settings from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM (NVIDIA NIM)
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL    = os.getenv("NVIDIA_MODEL", "openai/gpt-oss-20b")

# Vector DB (Pinecone)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "vector")

# Web Search (Tavily)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Database (Neon PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")

# MCP Server
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "https://deploy-9ugq.onrender.com/mcp")

# Observability (LangSmith)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "langgraph-agent")
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false")

# Agent Tuning
MAX_ITERATIONS    = int(os.getenv("MAX_ITERATIONS", "3"))
RETRIEVER_K       = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "20"))

# RAG / Embeddings
# EMBEDDING_PROVIDER: local | nvidia | openai   (see rag/embeddings.py)
#   "local" runs sentence-transformers/all-MiniLM-L6-v2 on CPU and needs no API
#   key. It emits 384-dim vectors, so it is NOT compatible with an index built
#   for the reference project's 1536-dim placeholder. Create a new index when
#   switching, or retrieval will fail with a dimension error.
EMBEDDING_PROVIDER  = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL")  # None => provider default

# VECTOR_STORE: memory | pinecone
#   "memory" is the default so tests and local runs need no credentials.
#   "pinecone" requires PINECONE_INDEX and PINECONE_API_KEY.
VECTOR_STORE = os.getenv("VECTOR_STORE", "memory")

# Minimum cosine similarity for a chunk to count as evidence. Below this the
# agent is told the documents do not cover the topic rather than being given a
# weak near-miss. Measured against all-MiniLM-L6-v2: on-topic paraphrases score
# ~0.49-0.73, unrelated queries score <0.05.
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.30"))


# Validation
def validate(require_hosted: bool = False):
    """Check required env vars.

    The reference version raised at import time on four variables, which made
    ``import main`` impossible without full production credentials and stopped
    the test suite from even being collected (audit L6). Hosted credentials are
    therefore only required when a hosted provider is actually selected; the
    default local embedding path needs none of them.

    Parameters
    ----------
    require_hosted:
        Force the hosted keys to be present, regardless of provider.
    """
    required: dict[str, object] = {"DATABASE_URL": DATABASE_URL}

    if EMBEDDING_PROVIDER.lower() == "openai":
        required["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    else:
        required["NVIDIA_API_KEY"] = NVIDIA_API_KEY

    if VECTOR_STORE.lower() == "pinecone":
        required["PINECONE_API_KEY"] = PINECONE_API_KEY
        required["PINECONE_INDEX"] = PINECONE_INDEX

    if require_hosted:
        required["TAVILY_API_KEY"] = TAVILY_API_KEY
        if EMBEDDING_PROVIDER.lower() != "openai":
            required["NVIDIA_API_KEY"] = NVIDIA_API_KEY

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}. Copy .env.example to .env")
