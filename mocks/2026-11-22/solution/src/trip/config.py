from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/trip/config.py → trip-lab/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",          # tolère d'autres variables dans le .env
    )

    mistral_api_key: SecretStr                 # lit MISTRAL_API_KEY (casse ignorée)
    mistral_model: str = "mistral-large-latest"
    http_timeout_s: float = 5.0
    forecast_horizon_days: int = 16


@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore[call-arg]