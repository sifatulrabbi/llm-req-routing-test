import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def llm_as_router(q: str) -> tuple[str, float]:
    chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Your job is identify the best agent to handover the request to.\n"
                        "The agents that are currently available: 'skilled_agent', 'fast_agent', 'cost_effective_agent'\n"
                        "Here the 'skilled_agent' is the most skilled workhorse that can handle complex and multi-step tasks, but it is not the fastest. The 'fast_agent' is the fastest but average skilled agent to get a job done. The 'cost_effective_agent' is the most cost effective, but it is not the most skilled.\n"
                        "You MUST return only the name of the agent to handover the request to and nothing else."
                    ),
                ),
                ("user", "Here is the query from the user:\n\n{q}"),
            ]
        )
        | ChatOpenAI(model="gpt-4.1-mini", use_responses_api=True)
        | StrOutputParser()
    )
    start = time.time()
    response = chain.invoke({"q": q})
    consumed_time = time.time() - start
    print(f"  llm_as_router: {consumed_time:.2f} seconds")
    return response, consumed_time


if __name__ == "__main__":
    q = input("Enter a query: ")
    print("Chosen model: ", llm_as_router(q))
