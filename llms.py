"""LLM instances — imported across all nodes.

Embeddings deliberately live elsewhere. The reference implementation kept a
``MockEmbeddings`` class here that returned all-zero vectors and was wired
directly into the retrieval path, which made every similarity score identical
and therefore meaningless. Embedding providers are now selected by
``rag.embeddings.get_embeddings()``; see that module for the rationale and the
index-dimension implications.
"""
from langchain_openai import ChatOpenAI
import config

_base = dict(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL)

answer_LLM = ChatOpenAI(**_base)
router_LLM = ChatOpenAI(**_base, temperature=0)

# Separate vision-capable model — kept apart from answer_LLM because
# answer_LLM is also used for tool-calling (nodes/llm_tool.py), and not
# all NVIDIA NIM models support both function-calling and vision at once.
vision_LLM = ChatOpenAI(
    base_url=config.NVIDIA_BASE_URL,
    api_key=config.NVIDIA_API_KEY,
    model="meta/llama-3.2-90b-vision-instruct",
)
