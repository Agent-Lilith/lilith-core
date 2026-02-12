from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class BaseAgentSettings(BaseSettings):
    DATABASE_URL: str = "postgresql://lilith:lilith@localhost:5432/lilith"
    EMBEDDING_URL: str = "http://localhost:8080"
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )
