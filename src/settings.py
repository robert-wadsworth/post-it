from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4o-mini"
    image_model: str = "dall-e-2"
    image_size: str = "1024x1024"
    max_revisions: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
