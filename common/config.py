"""
Shared settings for all Lilith agent projects (lilith-emails, lilith-whatsapp, etc.).

Projects should subclass BaseAgentSettings and add only project-specific fields.
When you run a project (e.g. `uv run python main.py` from lilith-emails), the
.env file loaded is the *project's* .env (current working directory), not this
package's .env. Copy shared vars from this package's .env.example into each
project's .env. After updating lilith-core, run `uv sync` in each project so
the subclass picks up new base fields.
"""

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
