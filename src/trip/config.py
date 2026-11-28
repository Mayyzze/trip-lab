from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    # A name of trip.extract.VARIANTS (a test checks the list). Chosen in S5 from
    # scripts/compare_prompts.py, measured on mistral-small-latest.
    extraction_variant: Literal["zero_shot", "few_shot", "cot", "structured"] = "few_shot"
    http_timeout_s: float = 5.0
    forecast_horizon_days: int = 16
    weather_cache_enabled: bool = True
    forecast_cache_ttl_s: float = 3600
    embed_model: str = "mistral-embed"
    embed_batch_size: int = 32
    chunk_size: int = 800                      # characters
    chunk_overlap: int = 100                   # characters
    rag_top_k: int = 4
    guides_dir: Path = PROJECT_ROOT / "data" / "guides"
    index_dir: Path = PROJECT_ROOT / "data" / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings() # type: ignore[call-arg]