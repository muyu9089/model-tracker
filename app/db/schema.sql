CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    first_seen_at TEXT NOT NULL,
    published_at TEXT,
    title TEXT,
    source_url TEXT,
    aihot_url TEXT,
    aihot_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE model_profiles (
    model_id INTEGER PRIMARY KEY REFERENCES models(id) ON DELETE CASCADE,
    open_source_attribute TEXT,
    vendor_or_organization TEXT,
    model_series TEXT,
    release_or_api_url TEXT,
    release_date TEXT,
    model_architecture TEXT,
    region TEXT,
    bank_applicable_scenarios TEXT,
    minimum_gpu_and_count TEXT,
    minimum_context_length TEXT,
    minimum_concurrency TEXT,
    recommended_gpu_and_count TEXT,
    recommended_context_length TEXT,
    recommended_concurrency TEXT,
    license TEXT,
    total_parameters_b TEXT,
    active_parameters_b TEXT,
    input_modalities TEXT,
    output_modalities TEXT,
    collected_at TEXT NOT NULL
);

CREATE TABLE benchmarks (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL
);

CREATE TABLE benchmark_states (
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id),
    status TEXT NOT NULL CHECK (status IN ('pending', 'recorded', 'failed')),
    update_error TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (model_id, benchmark_id)
);

CREATE TABLE benchmark_results (
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id),
    score_value REAL,
    score_unit TEXT,
    output_tokens INTEGER,
    time_seconds REAL,
    matched_model_name TEXT,
    matched_slug TEXT,
    matched_parameters_b REAL,
    matched_reasoning_depth TEXT,
    match_method TEXT,
    source_url TEXT,
    collected_at TEXT NOT NULL,
    PRIMARY KEY (model_id, benchmark_id)
);

CREATE TABLE source_versions (
    id INTEGER PRIMARY KEY,
    source_code TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content_text TEXT NOT NULL,
    detected_at TEXT NOT NULL
);

CREATE TABLE scan_runs (
    id INTEGER PRIMARY KEY,
    mode TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    new_model_count INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT
);

CREATE TABLE scan_run_models (
    run_id INTEGER NOT NULL REFERENCES scan_runs(id) ON DELETE CASCADE,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    discovery_order INTEGER NOT NULL,
    PRIMARY KEY (run_id, model_id)
);
