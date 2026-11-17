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
    review: bool = Field(
        False,
        description=(
            "True when a human should look at this message. It is separate "
            "from the action on purpose: the escalation ladder can raise a "
            "player to 'flag for review' while the message itself is held "
            "back from other players, and that combination still has to "
            "reach a moderator."
        ),
    )
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


class CompressRequest(BaseModel):
    """A payload to compress.

    ``content_type`` forces a compressor instead of auto-detecting one.
    ``query`` biases query-aware compressors toward the parts of the payload
    that matter to the question being asked.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        max_length=262_144,
        description="The payload to compress.",
    )
    content_type: str | None = Field(
        None,
        max_length=64,
        description="Force a compressor (json, log, tabular, config, ...) "
        "instead of auto-detecting one.",
    )
    query: str = Field(
        "",
        max_length=1024,
        description="Bias what survives toward this query.",
    )
    mode: Literal["compress", "record"] = Field(
        "compress",
        description="'record' never transforms; it only reports the token counts.",
    )


class CompressResponse(BaseModel):
    """The compressed payload and the accounting behind it.

    Every token figure is a local estimate, not provider usage. ``basis``
    says which, and it is always ``inferred`` — the engine never claims a
    measured number it did not measure.
    """

    output: str = Field(..., description="The compressed payload.")
    content_type: str = Field(..., description="The detected or forced content type.")
    tokens_before: int = Field(..., description="Estimated tokens in the input.")
    tokens_after: int = Field(..., description="Estimated tokens in the output.")
    tokens_saved: int = Field(..., description="Estimated tokens removed.")
    token_count_basis: str = Field(..., description="Which estimator produced the counts.")
    ratio: float = Field(..., description="Fraction of tokens removed, 0..1.")
    basis: str = Field(..., description="Always 'inferred' for the token figures.")
    recovery_handle: str | None = Field(
        None,
        description="Handle for byte-exact recovery of the original. "
        "Present only when the payload was actually transformed.",
    )
    method: str | None = Field(
        None, description="Which compressor ran. Absent on pass-through."
    )
    lossless_to_model: bool | None = Field(
        None,
        description="Whether the output kept the full value without data dropped.",
    )
    passed_through: bool = Field(
        ...,
        description="True when the input was returned unchanged and nothing was claimed.",
    )


class RecoverResponse(BaseModel):
    """The exact original bytes for a recovery handle."""

    handle: str
    content: str = Field(..., description="The original payload, byte-for-byte.")
    content_type: str = Field(..., description="The content type it was stored under.")
