from sqlalchemy import create_engine, Engine
from app.config import Config


def get_duckdb_engine(database_path: str = None) -> Engine:
    if database_path:
        # If absolute path or starts with data/, use as is.
        # But usually we want a relative path to be treated as a file.
        # duckdb:///path/to/file.db -> relative to CWD

        # If it's an in-memory request specifically
        if database_path == ":memory:":
            return create_engine("duckdb:///:memory:")

        return create_engine(f"duckdb:///{database_path}")

    config = Config()
    return create_engine(config.database_url)
