from pathlib import Path

from .connection import transaction


SCHEMA_VERSION = 4


def migrate(path: Path) -> None:
    with transaction(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"数据库版本 {version} 高于程序支持的版本 {SCHEMA_VERSION}")
        if version == 0:
            schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            connection.executescript(schema)
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version == 1:
            connection.execute("""CREATE TABLE aa_model_mappings (
                model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id),
                slug TEXT NOT NULL,
                PRIMARY KEY (model_id, benchmark_id)
            )""")
            connection.execute("PRAGMA user_version = 2")
            version = 2
        if version == 2:
            connection.execute("ALTER TABLE models ADD COLUMN model_info_updated_at TEXT")
            connection.execute("""UPDATE models SET model_info_updated_at = (
                SELECT collected_at FROM model_profiles WHERE model_id = models.id
            ) WHERE EXISTS (SELECT 1 FROM model_profiles WHERE model_id = models.id)""")
            connection.execute("PRAGMA user_version = 3")
            version = 3
        if version == 3:
            connection.execute("""CREATE TABLE aa_model_mappings_new (
                model_id INTEGER PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
                slug TEXT NOT NULL
            )""")
            connection.execute("""INSERT INTO aa_model_mappings_new(model_id, slug)
                SELECT old.model_id, old.slug FROM aa_model_mappings old
                WHERE old.benchmark_id = (SELECT MIN(other.benchmark_id)
                    FROM aa_model_mappings other WHERE other.model_id = old.model_id)""")
            connection.execute("DROP TABLE aa_model_mappings")
            connection.execute("ALTER TABLE aa_model_mappings_new RENAME TO aa_model_mappings")
            connection.execute("""UPDATE benchmark_states SET status='pending'
                WHERE model_id IN (SELECT model_id FROM aa_model_mappings)""")
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
