"""
Configuration module for KisanSathi backend.
Loads settings from environment variables or .env file with pydantic-settings.
"""

from typing import List, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General App Settings
    APP_NAME: str = "KisanSathi"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Regional & Agronomic Scoping (v1 default: Maharashtra, 4 crops, Hindi + Telugu)
    DEFAULT_STATE: str = "Maharashtra"
    DEFAULT_LANGUAGE: str = "hi"
    SECONDARY_LANGUAGE: str = "te"  # Configured regional language: Telugu
    SUPPORTED_LANGUAGES: str = "en,hi,te"
    SUPPORTED_CROPS: str = "tomato,onion,soybean,cotton"

    # RAG Configuration
    CHUNKING_STRATEGY: Literal["fixed", "recursive", "semantic"] = "recursive"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.58
    VECTOR_STORE_BACKEND: Literal["inmemory", "chroma", "qdrant"] = "inmemory"
    EMBEDDING_PROVIDER: Literal["mock", "sentence-transformers", "bge-m3", "openai"] = "mock"

    # LLM Configuration
    LLM_PROVIDER: Literal["mock", "gemini", "openai", "ollama"] = "mock"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "gemini-1.5-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512

    # Speech Configuration
    STT_PROVIDER: str = "mock"
    TTS_PROVIDER: str = "mock"

    # Tools Configuration
    OPENWEATHER_API_KEY: str = ""
    DATA_GOV_IN_API_KEY: str = ""
    CACHE_TTL_SECONDS: int = 1800

    # Security & Guardrails
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "adminpassword123"
    JWT_SECRET: str = "super_secret_jwt_key_for_kisansathi"
    RATE_LIMIT_PER_MINUTE: int = 60
    ENABLE_SAFETY_GUARDRAILS: bool = True

    # Escalation
    KISAN_CALL_CENTRE_HELPLINE: str = "1800-180-1551"

    @property
    def crop_list(self) -> List[str]:
        return [c.strip().lower() for c in self.SUPPORTED_CROPS.split(",") if c.strip()]

    @property
    def language_list(self) -> List[str]:
        return [l.strip().lower() for l in self.SUPPORTED_LANGUAGES.split(",") if l.strip()]


# Global singleton instance
settings = Settings()
