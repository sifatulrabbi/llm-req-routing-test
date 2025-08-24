CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS queries (
  id bigserial PRIMARY KEY,
  query_text text NOT NULL,
  embedding vector(256) NOT NULL,
  model_type text NOT NULL
);

CREATE INDEX IF NOT EXISTS queries_embedding_idx
ON queries USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);