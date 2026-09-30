"""Request and response schemas for the HTTP API.

Extends the reference project's two models. Changes worth noting:

- ``query`` now has a length bound. The reference validated only "not empty",
  so a caller could submit a multi-megabyte prompt and be billed for it
  (audit L2).
- The response carries routing and provenance, so a client can see which
  route was chosen and what was cited without a second request. The reference
  returned only ``{"response": "..."}``.
"""

from __future__ import annotations

import base64
import binascii

from pydantic import BaseModel, Field, field_validator

from app.core.config import Settings

MAX_IMAGE_BASE64_CHARS = 5_000_000


class ChatRequest(BaseModel):
    """Incoming chat request."""

    query: str = Field(
        min_length=1,
        max_length=4000,
        description="The user's question.",
    )
    thread_id: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Conversation thread. Omit for a fresh conversation. Note the "
            "reference project defaulted this to \"1\", which made every "
            "caller that omitted it share one global conversation."
        ),
    )
    image_base64: str | None = Field(
        default=None,
        description="Optional data URI or raw base64 image.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty or whitespace only")
        return v

    @field_validator("thread_id")
    @classmethod
    def thread_id_sane(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        # thread_id is interpolated into LangGraph config and used as a
        # Postgres partition key; keep it to a conservative character set.
        if not all(c.isalnum() or c in "-_" for c in cleaned):
            raise ValueError(
                "thread_id may only contain letters, digits, hyphen, underscore"
            )
        return cleaned

    @field_validator("image_base64")
    @classmethod
    def image_base64_valid(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) > MAX_IMAGE_BASE64_CHARS:
            raise ValueError(
                f"image_base64 too large ({len(v)} chars, max {MAX_IMAGE_BASE64_CHARS})"
            )
        payload = v.split(",", 1)[1] if v.startswith("data:") else v
        if not payload:
            raise ValueError("image_base64 payload is empty")
        try:
            base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_base64 is not valid base64: {exc}")
        return v

    def resolved_thread_id(self, settings: Settings) -> str:
        return self.thread_id or f"anon-{settings.default_thread_id}"


class Source(BaseModel):
    """A citation returned alongside an answer."""

    filename: str
    page_number: int | None = None
    chunk_id: str
    score: float


class ChatResponse(BaseModel):
    """Assistant reply plus provenance."""

    response: str
    thread_id: str
    request_id: str
    route: str | None = None
    route_confidence: float | None = None
    grounded: bool | None = None
    insufficient_information: bool | None = None
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    iterations: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    embeddings: str
    vector_store: str
    memory: str
