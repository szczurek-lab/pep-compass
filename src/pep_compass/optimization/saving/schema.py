"""Canonical relational run schema shared by memory storage and SQLite files."""

FORMAT_VERSION = 2

SCHEMA = """
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
    settings_sha256 TEXT NOT NULL, summary_json TEXT NOT NULL
);
CREATE TABLE iterations (
    iteration_id INTEGER PRIMARY KEY,
    loop_execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    parent_iteration_id INTEGER REFERENCES iterations(iteration_id),
    node_id TEXT NOT NULL, iteration_index INTEGER NOT NULL,
    scope_json TEXT NOT NULL, trajectories INTEGER,
    elapsed_seconds REAL, duration_seconds REAL, interval_seconds REAL,
    step_duration_sum_seconds REAL,
    memory_start_bytes INTEGER, memory_peak_bytes INTEGER, memory_end_bytes INTEGER,
    input_count INTEGER, output_count INTEGER, status TEXT NOT NULL
);
CREATE TABLE step_executions (
    execution_id INTEGER PRIMARY KEY, node_id TEXT NOT NULL,
    parent_execution_id INTEGER REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),
    input_batch_id INTEGER REFERENCES batches(batch_id),
    output_batch_id INTEGER REFERENCES batches(batch_id),
    scope_json TEXT NOT NULL, input_count INTEGER, output_count INTEGER,
    duration_seconds REAL, memory_start_bytes INTEGER,
    memory_peak_bytes INTEGER, memory_end_bytes INTEGER,
    status TEXT NOT NULL, phase TEXT NOT NULL, error TEXT
);
CREATE TABLE execution_metrics (
    metric_id INTEGER PRIMARY KEY,
    execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),
    stage_name TEXT NOT NULL, duration_seconds REAL,
    memory_start_bytes INTEGER, memory_peak_bytes INTEGER, memory_end_bytes INTEGER,
    input_count INTEGER, output_count INTEGER, status TEXT NOT NULL,
    UNIQUE(execution_id,stage_name)
);
CREATE TABLE peptides (peptide_id INTEGER PRIMARY KEY, sequence TEXT NOT NULL UNIQUE);
CREATE TABLE candidate_instances (
    candidate_id INTEGER PRIMARY KEY, peptide_id INTEGER REFERENCES peptides(peptide_id)
);
CREATE TABLE batches (
    batch_id INTEGER PRIMARY KEY,
    producer_execution_id INTEGER REFERENCES step_executions(execution_id),
    candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0)
);
CREATE TABLE batch_members (
    batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    position INTEGER NOT NULL CHECK(position >= 0),
    candidate_id INTEGER NOT NULL REFERENCES candidate_instances(candidate_id),
    PRIMARY KEY(batch_id,position)
);
CREATE TABLE batch_lineage (
    lineage_id INTEGER PRIMARY KEY,
    child_batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    child_start INTEGER NOT NULL, child_end INTEGER NOT NULL,
    parent_batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    parent_start INTEGER NOT NULL, parent_end INTEGER NOT NULL,
    mapping_kind TEXT NOT NULL CHECK(mapping_kind IN ('pairwise','broadcast')),
    execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),
    CHECK(child_start >= 0 AND child_end > child_start),
    CHECK(parent_start >= 0 AND parent_end > parent_start),
    CHECK((mapping_kind='pairwise' AND child_end-child_start=parent_end-parent_start)
       OR (mapping_kind='broadcast' AND parent_end-parent_start=1))
);
CREATE TABLE objectives (
    objective TEXT PRIMARY KEY, direction TEXT NOT NULL
        CHECK(direction IN ('minimize','maximize')),
    configuration_sha256 TEXT
);
CREATE TABLE evaluations (
    evaluation_id INTEGER PRIMARY KEY,
    objective TEXT NOT NULL REFERENCES objectives(objective),
    peptide_id INTEGER NOT NULL REFERENCES peptides(peptide_id),
    candidate_id INTEGER REFERENCES candidate_instances(candidate_id),
    execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),
    evaluation_index INTEGER NOT NULL UNIQUE, score REAL NOT NULL
);
CREATE TABLE tensor_payloads (
    payload_id TEXT PRIMARY KEY, codec_version INTEGER NOT NULL, data BLOB NOT NULL
);
CREATE TABLE batch_payloads (
    batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    field_name TEXT NOT NULL,
    payload_id TEXT NOT NULL REFERENCES tensor_payloads(payload_id),
    PRIMARY KEY(batch_id,field_name)
);
CREATE TABLE candidate_sets (
    set_id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    node_id TEXT, execution_id INTEGER REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id), kind TEXT NOT NULL
);
CREATE TABLE component_records (
    record_id INTEGER PRIMARY KEY, node_id TEXT,
    execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    name TEXT NOT NULL, payload_id TEXT NOT NULL REFERENCES tensor_payloads(payload_id)
);
CREATE TABLE checkpoints (
    checkpoint_id INTEGER PRIMARY KEY, node_id TEXT NOT NULL,
    execution_id INTEGER NOT NULL REFERENCES step_executions(execution_id),
    iteration_id INTEGER REFERENCES iterations(iteration_id),
    scope_json TEXT NOT NULL, phase TEXT NOT NULL,
    payload_id TEXT NOT NULL REFERENCES tensor_payloads(payload_id),
    output_count INTEGER, output_sha256 TEXT
);
CREATE INDEX evaluations_objective_peptide ON evaluations(objective,peptide_id,evaluation_index);
CREATE INDEX execution_iteration ON step_executions(iteration_id,parent_execution_id);
CREATE INDEX execution_parent ON step_executions(parent_execution_id);
CREATE INDEX iteration_parent ON iterations(parent_iteration_id);
CREATE INDEX execution_node ON step_executions(node_id,execution_id);
CREATE INDEX lineage_child ON batch_lineage(child_batch_id,child_start);
CREATE INDEX member_candidate ON batch_members(candidate_id,batch_id);
CREATE INDEX checkpoints_node ON checkpoints(node_id,execution_id);
CREATE VIEW evaluation_records AS
 SELECT e.evaluation_id,e.objective,p.sequence,e.score,o.direction,
        e.evaluation_index,e.candidate_id,e.execution_id,e.iteration_id
 FROM evaluations e JOIN peptides p USING(peptide_id)
 JOIN objectives o USING(objective);
CREATE VIEW evaluation_archive AS
 SELECT r.* FROM evaluation_records r
 WHERE r.evaluation_index=(
   SELECT MAX(e.evaluation_index) FROM evaluations e
   JOIN peptides p USING(peptide_id)
   WHERE e.objective=r.objective AND p.sequence=r.sequence
 );
"""
