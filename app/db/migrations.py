from pathlib import Path

from .connection import transaction


SCHEMA_VERSION = 1


def migrate(path: Path) -> None:
    with transaction(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"数据库版本 {version} 高于程序支持的版本 {SCHEMA_VERSION}")
        if version == 0:
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            connection.executescript(schema)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
