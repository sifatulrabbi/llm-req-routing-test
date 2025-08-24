import time
import os
import re
import psycopg
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector, Vector


load_dotenv()


EMBEDDING_MODEL = "text-embedding-3-large"


def _safe_ident(identifier: str) -> str:
    """Validate and return a safe SQL identifier (table/column name)."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier or ""):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def _build_dsn_from_env() -> str:
    """Build a PostgreSQL DSN from environment variables.

    Respects `DATABASE_URL` if present. Otherwise uses PGHOST, PGPORT, PGDATABASE,
    PGUSER, and PGPASSWORD.
    """
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return dsn

    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "postgres"
    user = os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or os.getenv("USER")
    password = os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or ""

    parts: List[str] = [f"host={host}", f"port={port}", f"dbname={dbname}"]
    if user:
        parts.append(f"user={user}")
    if password:
        parts.append(f"password={password}")
    return " ".join(parts)


def _embed_text(text: str) -> List[float]:
    """Create an embedding vector for the provided text using OpenAI."""
    if not text:
        return []
    client = OpenAI()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    # OpenAI returns a list with one embedding for single input
    vec: List[float] = resp.data[0].embedding  # type: ignore[no-any-return]
    # Ensure dimensionality matches the DB column (vector(256))
    target_dim = 256
    if len(vec) >= target_dim:
        vec = vec[:target_dim]
    else:
        vec = vec + [0.0] * (target_dim - len(vec))
    # Normalize to unit length to align with cosine distance expectations
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def manual_router(q: str) -> tuple[str, float]:
    """Perform a vector similarity search against PostgreSQL using pgvector.

    The query uses cosine distance via the `<->` operator on a `vector` column.
    Configure table/column names with env vars (optional):
    - PGVECTOR_TABLE (default: queries)
    - PGVECTOR_EMBEDDING_COLUMN (default: embedding)
    - PGVECTOR_TEXT_COLUMN (default: query_text)
    - PGVECTOR_ID_COLUMN (default: id)

    Returns the top result as a dictionary, or None if no results.
    """
    if not q:
        return None

    table = _safe_ident(os.getenv("PGVECTOR_TABLE", "queries"))
    embedding_col = _safe_ident(os.getenv("PGVECTOR_EMBEDDING_COLUMN", "embedding"))
    text_col = _safe_ident(os.getenv("PGVECTOR_TEXT_COLUMN", "query_text"))
    model_type_col = _safe_ident(os.getenv("PGVECTOR_MODELTYPE_COLUMN", "model_type"))
    id_col = _safe_ident(os.getenv("PGVECTOR_ID_COLUMN", "id"))

    dsn = _build_dsn_from_env()
    sql = (
        f"SELECT {id_col}, {text_col}, {model_type_col}, {embedding_col} <-> %s AS distance "
        f"FROM {table} "
        f"ORDER BY {embedding_col} <-> %s "
        f"LIMIT %s"
    )

    # Connect and execute query
    try:
        with psycopg.connect(dsn) as conn:
            # Ensure pgvector adapter is registered for this connection
            register_vector(conn)
            with conn.cursor() as cur:
                start = time.time()
                query_embedding = _embed_text(q)
                if not query_embedding:
                    return None
                vector_param = Vector(query_embedding)

                cur.execute(sql, (vector_param, vector_param, 1))
                row = cur.fetchone()
                if not row:
                    return None
                result = {
                    "id": row[0],
                    "text": row[1],
                    "model_type": row[2],
                    "distance": row[3],
                }

                consumed_time = time.time() - start
                print(f"  manual_router: {consumed_time:.2f} seconds")
                return result["model_type"], consumed_time
    except Exception:
        # Prefer returning None on operational errors for a simple API surface
        # Consider logging in a real application
        return ""


if __name__ == "__main__":
    q = input("Enter a query: ")
    print("Chosen model: ", manual_router(q))
