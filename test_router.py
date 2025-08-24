from llm_as_router import llm_as_router
from manual_router import manual_router


queries: list[tuple[str, str]] = [
    (
        "extract all the requirements and prepare a plan on how we can write a proposal for the requirements",
        "skilled_agent",
    ),
    (
        "hello, write me structure of how a perfect proposal should look like",
        "skilled_agent",
    ),
    (
        'I need a new proposal with filename "PRoposal xyz - draft 02" and the language should be EN. Now come up with a plan for that.',
        "skilled_agent",
    ),
    (
        "Summarize the last internal newsletter.",
        "cost_effective_agent",
    ),
    (
        "list down the security measures of sifatul's lab",
        "fast_agent",
    ),
    (
        "Find all files related to the annual report.",
        "cost_effective_agent",
    ),
    (
        "What are the company's policies on intellectual property?",
        "cost_effective_agent",
    ),
]


failed_manual_queries = []
failed_llm_as_router_queries = []
total_time_manual = 0
total_time_llm_as_router = 0

for q in queries:
    res, time = llm_as_router(q[0])
    total_time_llm_as_router += time
    if q[1] != res:
        failed_llm_as_router_queries.append((q[0], q[1], res))

    res, time = manual_router(q[0])
    total_time_manual += time
    if q[1] != res:
        failed_manual_queries.append((q[0], q[1], res))

print()
print(
    f"average time for llm_as_router: {total_time_llm_as_router / len(queries):.2f} seconds"
)
print(f"average time for manual_router: {total_time_manual / len(queries):.2f} seconds")
print()
print("failed manual queries:")
for q in failed_manual_queries:
    print(f"FAILED: {q[0]} -> {q[1]} != {q[2]}")
print()
print("failed llm_as_router queries:")
for q in failed_llm_as_router_queries:
    print(f"FAILED: {q[0]} -> {q[1]} != {q[2]}")
