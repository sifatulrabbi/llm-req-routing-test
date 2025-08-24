import os
import math
import random
import psycopg
from dotenv import load_dotenv
from typing import List
from pgvector.psycopg import register_vector, Vector


load_dotenv()

cost_effective_agent_queries = [
    "Hello",
    "Hello who are you?",
    "What's the weather like today?",
    "Set a reminder for 6 PM.",
    "Translate 'hello' into Spanish.",
    "Tell me a joke.",
    "What time is it in New York?",
    "Solve 25 × 14.",
    "Summarize this text for me.",
    "Play relaxing music.",
    "Who is the president of the United States?",
    "Write me a short email to schedule a meeting.",
    "Summarize our conversation so far.",
    "Define 'synergy' in simple terms.",
    "Give me today's top business headline.",
    "Convert 50 USD to EUR.",
    "Set a timer for 10 minutes.",
    "What's on my calendar for tomorrow?",
    "Find synonyms for 'innovative'.",
    "Spell-check this sentence: 'The comittee has resieved your propsal.'",
    "What's 20% of 175?",
    "Explain this acronym: KPI.",
    "Give me a motivational quote for presentations.",
]
fast_agent_queries = [
    "Fix the grammar in this proposal paragraph.",
    "Convert this sentence into a slide title.",
    "Give me three bullet points from this text.",
    "What's the word count of this section?",
    "Turn this list into a numbered list.",
    "Extract all dates from this project plan.",
    "Summarize these meeting notes in 3 lines.",
    'Rename all "client" mentions to "partner".',
    "What's the total in this budget table?",
    "Format this JSON so it's readable.",
    "Extract all action items from this meeting transcript.",
    "Turn these notes into a checklist.",
    "Highlight all numbers in this report.",
    "Summarize this paragraph into one sentence.",
    "Generate 3 subject line options from this email draft.",
    "Convert this paragraph into plain English for a client.",
    "Find duplicate entries in this list.",
    "Format this table into CSV.",
    "Rewrite this sentence to sound more formal.",
    "Create a one-line caption for this chart.",
]
skilled_agent_queries = [
    "Compare the strengths and weaknesses of our current proposal draft against the client's RFP.",
    "Create a 5-slide narrative that explains why our solution is better than the competitor's.",
    "Analyze this project timeline and suggest where we can cut 2 weeks without increasing costs.",
    "Audit this budget sheet and identify inconsistencies or risks.",
    "Based on last quarter's marketing metrics, recommend the 3 most effective channels to prioritize.",
    "Review this presentation and suggest how to make the storyline more persuasive.",
    "Read through this compliance document and highlight where our policies may fall short.",
    "Generate potential objections a client might raise to this proposal, and give rebuttals.",
    "Evaluate this code snippet for performance bottlenecks and propose improvements.",
    "Propose a structured rollout plan for migrating our app to a microservices architecture.",
    "Draft a client proposal based on this requirements document.",
    "Design a project plan with milestones for launching a new feature in 3 months.",
    "Analyze customer feedback and categorize top 5 recurring issues.",
    "Suggest improvements to our pitch deck to appeal to investors.",
    "Review this audit log and identify potential security risks.",
    "Based on these sales numbers, forecast next quarter's revenue.",
    "Compare these two marketing strategies and recommend one.",
    "Outline risks and mitigation strategies for this migration project.",
    "Identify gaps in this proposal that might cause client pushback.",
    "Create an executive summary of this 20-page technical report.",
]


def build_dsn_from_env() -> str:
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


def generate_random_unit_vector(dim: int = 256, seed: int | None = None) -> List[float]:
    if seed is not None:
        rnd = random.Random(seed)
        values = [rnd.gauss(0.0, 1.0) for _ in range(dim)]
    else:
        values = [random.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def seed_queries():
    # Curated queries labeled with the appropriate agent type
    # Allowed values: "skilled_agent", "fast_agent", "cost_effective_agent"
    allowed_types = {"skilled_agent", "fast_agent", "cost_effective_agent"}

    # Combine all queries with their appropriate model types
    all_queries = []

    # Add cost-effective agent queries
    for query in cost_effective_agent_queries:
        all_queries.append((query, "cost_effective_agent"))

    # Add fast agent queries
    for query in fast_agent_queries:
        all_queries.append((query, "fast_agent"))

    # Add skilled agent queries
    for query in skilled_agent_queries:
        all_queries.append((query, "skilled_agent"))

    dsn = build_dsn_from_env()
    inserted = 0
    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            rows = []
            # Make embeddings reproducible across runs
            base_seed = 424242
            for i, (text, agent_type) in enumerate(all_queries):
                if agent_type not in allowed_types:
                    raise ValueError(f"Invalid model_type: {agent_type}")
                embedding = Vector(generate_random_unit_vector(256, seed=base_seed + i))
                rows.append((text, embedding, agent_type))

            cur.executemany(
                """
                INSERT INTO queries (query_text, embedding, model_type)
                VALUES (%s, %s, %s)
                """,
                rows,
            )
            inserted = cur.rowcount or 0
        conn.commit()

    print(f"Inserted {inserted} rows into queries table.")
    print(
        f"Breakdown: {len(cost_effective_agent_queries)} cost-effective, {len(fast_agent_queries)} fast, {len(skilled_agent_queries)} skilled agent queries"
    )


if __name__ == "__main__":
    seed_queries()
