"""Request and response contracts for the moderation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Action = Literal["allow", "flag", "block", "shadow_mute"]
Escalation = Literal["none", "warn", "flag", "mute", "ban_review"]

# Player identifiers are interpolated into Redis keys, so they are restricted
# to characters that cannot escape the key namespace.
PLAYER_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"

# Channels are only logged, but they are attacker-controlled too, so they get
# the same restriction as the identifiers.
CHANNEL_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ModerateRequest(BaseModel):
    """A single chat message submitted for moderation.

    ``target_id`` is who the message is aimed at (a DM recipient or a
    mentioned player). ``is_report`` marks the submission as a player report
    *about* ``target_id``, filed by ``player_id``.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., max_length=4096, description="Raw chat message exactly as sent.")
    player_id: str = Field(..., pattern=PLAYER_ID_PATTERN)
    target_id: str | None = Field(None, pattern=PLAYER_ID_PATTERN)
    channel: str = Field("global", pattern=CHANNEL_PATTERN)
    account_age_s: int | None = Field(
        None, ge=0, description="Age of the sender account in seconds."
    )
    is_report: bool = Field(
        False, description="True when player_id is reporting target_id."
    )

    @field_validator("text")
    @classmethod
    def _reject_unencodable_text(cls, value: str) -> str:
        """Reject lone surrogates.

        JSON permits ``"\\ud800"`` and pydantic accepts it as a ``str``, but
        encoding it as UTF-8 raises. Letting that through turns a bad request
        into a 500 instead of a 422.
        """
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("message contains characters that cannot be encoded") from exc
        return value


class ModerateResponse(BaseModel):
    """The moderation decision plus the evidence behind it."""

    action: Action = Field(..., description="What the game server should do.")
    reasons: list[str] = Field(
        ..., description="Machine-readable evidence, e.g. 'threat:0.97'."
    )
    scores: dict[str, float] = Field(
        ..., description="Per-label probabilities; empty if the model was skipped."
    )
    player_score: int = Field(..., description="Behaviour score of player_id.")
    target_score: int | None = Field(
        None, description="Behaviour score of target_id, when one was given."
    )
    escalation: Escalation = Field(..., description="Sender's ladder position.")
    muted_for_s: int | None = Field(
        None, description="Length of the mute applied by this decision, if any."
    )
    latency_ms: float = Field(..., description="Server-side processing time.")


class HealthResponse(BaseModel):
    """Liveness/readiness details."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_id: str | None = None
    redis_ok: bool
    version: str
