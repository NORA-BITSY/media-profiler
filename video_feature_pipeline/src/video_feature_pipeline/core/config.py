from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    data_root: Path = Field(default=Path("./data"))
    videos_dir: Path = Field(default=Path("./data/videos"))
    db_dir: Path = Field(default=Path("./data/chroma_db"))
    sqlite_path: Path = Field(default=Path("./data/pipeline.db"))

    model_config = SettingsConfigDict(env_prefix="VFP_PATHS_")


class IngestionConfig(BaseSettings):
    ytdlp_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
    max_concurrent: int = 3
    download_timeout: int = 300

    model_config = SettingsConfigDict(env_prefix="VFP_INGESTION_")


class ExtractionConfig(BaseSettings):
    frame_sample_rate: float = 1.0
    keyframe_rate: float = 0.5
    mediapipe_complexity: Literal[0, 1, 2] = 1
    insightface_model: str = "buffalo_l"
    insightface_det_size: list[int] = [640, 640]
    vlm_model: str = "moondream2"
    vlm_host: str = "http://localhost:11434"
    vlm_timeout: int = 30
    max_keyframes: int = 20
    # Adaptive frame rate settings
    adaptive_frame_rate: bool = False
    base_frame_rate: float = 0.5
    high_confidence_frame_rate: float = 2.0
    identity_confidence_threshold: float = 0.8

    model_config = SettingsConfigDict(env_prefix="VFP_EXTRACTION_")


class SearchConfig(BaseSettings):
    default_top_k: int = 10
    face_similarity_threshold: float = 0.6
    semantic_similarity_threshold: float = 0.5
    spatial_weight: float = 0.3
    face_weight: float = 0.5
    semantic_weight: float = 0.2

    model_config = SettingsConfigDict(env_prefix="VFP_SEARCH_")


class HarvestConfig(BaseSettings):
    enabled: bool = False
    socq_api_url: str = "https://api.socq.io"
    socq_api_key: str = ""
    min_confidence: float = 0.85
    max_concurrent: int = 3
    auto_scrape_on_new_identity: bool = True

    model_config = SettingsConfigDict(env_prefix="VFP_HARVEST_")


class LoggingConfig(BaseSettings):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    file: Path = Path("./data/pipeline.log")

    model_config = SettingsConfigDict(env_prefix="VFP_LOGGING_")


class Settings(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    harvest: HarvestConfig = Field(default_factory=HarvestConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_nested_delimiter="__",
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings