"""Runtime configuration.

Every setting can be overridden with a ``MOD_``-prefixed environment variable,
e.g. ``MOD_BLOCK_THRESHOLD=0.65``. See ``.env.example`` for the full list.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder used only when the operator has not set MOD_API_KEY. Startup logs
# a warning when this value is still in use.
DEFAULT_API_KEY = "dev-insecure-key-change-me"


def api_key_fingerprint(api_key: str) -> str:
    """A short, non-reversible identifier for an API key.

    The service logs this at startup and the client can print the same
    value, so an operator can confirm that both ends hold the same key
    without the key itself appearing anywhere.
    """
    if not api_key:
        return "none"
    return "sha256:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


class Settings(BaseSettings):
    """Service settings. Immutable for the lifetime of the process."""

    model_config = SettingsConfigDict(
        env_prefix="MOD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service -----------------------------------------------------------
    api_key: str = DEFAULT_API_KEY
    require_api_key: bool = True
    # Escape hatch for the demo and for local development only. Without it the
    # service refuses to start on the well-known development key, because that
    # key is public in this repository.
    allow_insecure_default_key: bool = False
    min_api_key_length: int = 16
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    enable_demo: bool = True
    # Whether the raw message text is written to the decision log. The
    # moderation audit trail needs it; a deployment with its own redaction
    # upstream may turn it off.
    log_message_text: bool = True

    # --- model -------------------------------------------------------------
    model_dir: Path = Path("models")
    onnx_file: str = "model_int8.onnx"
    tokenizer_file: str = "tokenizer.json"
    labels_file: str = "model_meta.json"
    onnx_threads: int = 2
    max_seq_len: int = 128
    score_cache_size: int = 2048

    # --- request limits ----------------------------------------------------
    max_message_chars: int = 256
    max_body_bytes: int = 4096
    # The compression endpoint takes whole files, not chat messages, so it gets
    # its own body limit. It is still authenticated and rate-limited; only the
    # size cap differs.
    compress_max_body_bytes: int = 262_144

    # --- toxicity thresholds ----------------------------------------------
    # Severity is the strongest trait the classifier found, and every trait
    # counts - there is no hard/soft split and no dead zone. Two bands:
    #
    #   severity >= block_threshold  -> the message is rejected outright
    #   severity >= review_threshold -> it is held back from other players
    #
    # Below the review threshold the message is delivered unchanged.
    block_threshold: float = 0.70
    review_threshold: float = 0.50
    # A player report is a claim, not evidence, and it carries ten points, so
    # it needs strong evidence: the target's own messages in the exchange
    # must have been at least this toxic. This is deliberately separate from
    # the review threshold - reusing the action band here made a report
    # worth 10 points for any mildly rude message, which is how a report
    # becomes a weapon.
    report_corroboration_threshold: float = 0.90

    # --- spam --------------------------------------------------------------
    rate_limit_count: int = 5
    rate_limit_window_s: int = 10
    new_account_rate_limit: int = 3
    new_account_age_s: int = 3600
    duplicate_window_s: int = 60
    recent_messages_kept: int = 20
    recent_messages_window_s: int = 300
    near_duplicate_threshold: float = 0.90
    max_caps_ratio: float = 0.70
    min_caps_length: int = 8
    max_symbol_ratio: float = 0.50
    max_single_char_run: int = 15
    max_spam_message_chars: int = 200
    allowlisted_link_hosts: tuple[str, ...] = ("example.com",)

    # --- harassment --------------------------------------------------------
    pair_window_s: int = 3600
    pair_message_limit: int = 5
    unanswered_threshold: int = 4
    pileon_window_s: int = 60
    pileon_threshold: int = 3
    escalation_margin: float = 0.15
    report_points: int = 10
    reporter_trust_full_count: int = 3
    reporter_trust_half_count: int = 10
    # One reporter may contribute at most this many points to one target per
    # window, so a single account cannot drive a player up the ladder alone.
    report_points_cap: int = 10
    report_points_window_s: int = 3600

    # --- behaviour score ---------------------------------------------------
    # Points per event, from the handoff's scoring table. A message that
    # trips several classifier labels counts once, not once per label.
    points_rate_limit: int = 1
    points_spam: int = 2
    points_toxic_block: int = 5
    points_toxic_review: int = 2
    points_harassment: int = 10
    points_report: int = 10

    score_decay_interval_s: int = 3600
    score_decay_points: int = 1
    score_ttl_s: int = 86400
    score_warn_at: int = 5
    score_flag_at: int = 15
    score_mute_at: int = 30
    score_ban_review_at: int = 60
    mute_durations_s: tuple[int, ...] = (600, 3600, 86400)
    mute_ttl_s: int = 604800

    # --- API rate limiting -------------------------------------------------
    api_rate_limit_count: int = 200
    api_rate_limit_global_count: int = 1000
    api_rate_limit_window_s: int = 1
    # Failed authentication attempts per client, counted locally so that a
    # brute-force attempt is bounded even when Redis is unreachable.
    failed_auth_limit: int = 10
    failed_auth_window_s: int = 60

    # --- behaviour ---------------------------------------------------------
    # Skip the ONNX model when a cheaper check already decided the outcome.
    # This is the point of the ordered pipeline: most abuse never reaches the
    # classifier, so it never costs CPU.
    skip_model_on_spam: bool = True

    # --- caveman compression -----------------------------------------------
    # The recovery store for the compression endpoint. ":memory:" keeps
    # recovery process-local, which suits a container with a read-only
    # filesystem; point it at a writable path to make handles durable across
    # restarts.
    caveman_ccr_db: str = ":memory:"

    # --- retired settings --------------------------------------------------
    # These used to control the toxicity decision. The decision now follows
    # the trait severity (see app/traits.py), so nothing reads them. They are
    # still accepted, so an existing .env keeps loading, and startup warns
    # when one is set - a knob that silently does nothing is worse than one
    # that is gone.
    flag_threshold: float | None = None
    hard_labels: tuple[str, ...] | None = None
    soft_labels: tuple[str, ...] | None = None
    points_toxic_flag: int | None = None

    @field_validator(
        "hard_labels",
        "soft_labels",
        "allowlisted_link_hosts",
        "mute_durations_s",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept ``a,b,c`` as well as JSON arrays from the environment."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            if stripped.startswith("["):
                return value
            return tuple(part.strip() for part in stripped.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _check_thresholds(self) -> Settings:
        """The thresholds have to describe a usable ladder.

        A misconfigured threshold would silently reopen the gap this ladder
        exists to close: a review threshold above the block threshold leaves
        the review band unreachable, and a block threshold below it rejects a
        message without a single trait to point at.
        """
        for name in (
            "block_threshold",
            "review_threshold",
            "report_corroboration_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.review_threshold > self.block_threshold:
            raise ValueError(
                "review_threshold must not be above block_threshold "
                f"({self.review_threshold} > {self.block_threshold}); "
                "otherwise the review band is unreachable"
            )
        return self

    @property
    def retired_settings_used(self) -> list[str]:
        """The retired settings that are still set in the environment.

        They are accepted so the service starts, but they no longer steer the
        decision, so the operator is told rather than left with a knob that
        does nothing.
        """
        return [
            name
            for name in (
                "flag_threshold",
                "hard_labels",
                "soft_labels",
                "points_toxic_flag",
            )
            if name in self.model_fields_set and getattr(self, name) is not None
        ]

    @property
    def using_default_api_key(self) -> bool:
        return self.api_key == DEFAULT_API_KEY

    def api_key_problem(self) -> str | None:
        """Describe why the API key is unacceptable, or ``None`` if it is fine.

        The development key is public in this repository, so a deployment that
        forgets to set a real one is an open door. The service refuses to
        start rather than run with it.
        """
        if not self.require_api_key:
            return None
        if self.using_default_api_key:
            if self.allow_insecure_default_key:
                return None
            return (
                "MOD_API_KEY is still the well-known development key; set a real "
                "key, or MOD_ALLOW_INSECURE_DEFAULT_KEY=true for local use"
            )
        if len(self.api_key) < self.min_api_key_length:
            return f"MOD_API_KEY must be at least {self.min_api_key_length} characters"
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
