"""
Shared settings for all Lilith agent projects (lilith-emails, lilith-whatsapp, etc.).

Projects should subclass BaseAgentSettings and add only project-specific fields.
When you run a project (e.g. `uv run python main.py` from lilith-emails), the
.env file loaded is the *project's* .env (current working directory), not this
package's .env. Copy shared vars from this package's .env.example into each
project's .env. After updating lilith-core, run `uv sync` in each project so
the subclass picks up new base fields.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Keys that must exist on BaseAgentSettings; used by projects to fail fast if core is stale
SHARED_SETTINGS_KEYS = (
    "VLLM_URL",
    "SPACY_API_URL",
    "FASTTEXT_LANGDETECT_URL",
    "EMBEDDING_URL",
)


class BaseAgentSettings(BaseSettings):
    """Shared schema: DB, embedding, transform (vLLM/Spacy/FastText). Subclass and add project fields."""

    DATABASE_URL: str = "postgresql://lilith:lilith@localhost:5432/lilith"
    EMBEDDING_URL: str = "http://localhost:6003"
    LOG_LEVEL: str = "INFO"
    VLLM_URL: str = ""
    VLLM_MODEL: str = "Qwen3-8B-AWQ"
    SPACY_API_URL: str = ""
    FASTTEXT_LANGDETECT_URL: str = ""
    LILITH_SCORE_CALIBRATION_PATH: str = ".lilith_score_calibration.json"
    LILITH_SCORE_WINDOW_SIZE: int = 5000
    LILITH_SCORE_DRIFT_Z: float = 1.5
    LILITH_SCORE_RECENCY_HALF_LIFE_DAYS: float = 180.0
    LILITH_ENABLE_LEARNED_RANKING: bool = False
    LILITH_SOURCE_RELIABILITY_PRIORS: dict[str, float] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_base_settings() -> BaseAgentSettings:
    return BaseAgentSettings()
