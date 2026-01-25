from pydantic_settings import BaseSettings


class Config(BaseSettings):
    app_name: str = "My Application"
    debug_mode: bool = False
    database_url: str = "duckdb:///data/fraud.db"
    gemini_api_key: str = ""
    chromadb_api_key: str = ""

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project_name: str = "Default Project"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Config()
