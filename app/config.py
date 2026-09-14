from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aihot_actor: str | None = None
    aihot_mcp_base_url: str = "https://aihot.news/api/mcp"
    mcp_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.8-flash"
    qwen_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    qwen_temperature: float = Field(default=0.0, ge=0, le=2)
    qwen_enable_thinking: bool = False
    qwen_request_attempts: int = Field(default=3, ge=1, le=5)
    tracker_timezone: str = "Asia/Shanghai"
    tracker_day_of_week: str = "mon"
    tracker_hour: int = Field(default=9, ge=0, le=23)
    tracker_minute: int = Field(default=0, ge=0, le=59)
    artificial_analysis_enabled: bool = True
    artificial_analysis_base_url: str = "https://artificialanalysis.ai"
    artificial_analysis_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @property
    def aihot_mcp_url(self) -> str:
        if not self.aihot_actor:
            return self.aihot_mcp_base_url
        separator = "&" if "?" in self.aihot_mcp_base_url else "?"
        return f"{self.aihot_mcp_base_url}{separator}aihot_actor={self.aihot_actor}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
