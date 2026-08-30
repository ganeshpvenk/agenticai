# pip install openai-agents google-search-results python-dotenv

import os
import sys
from openai import OpenAI
from agents import Agent, Runner, function_tool
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.insert(0, os.path.dirname(__file__))
from support_ticketing_agent_hitl import retrieve  # reuse the same FAQ vector search for both approaches

MODEL = "gpt-4o-mini"
openai_client = OpenAI()

# =====================================================================
# Same comparison as traditional_vs_agentic_rag.py, but the live-web leg
# is backed by SerpAPI (Google search results) instead of Tavily. Both
# approaches still answer against the SAME FAQ knowledge base
# (support_ticketing_agent_hitl.py's ChromaDB index), so the comparison
# isolates the RETRIEVAL STRATEGY, not the data.
#
# Traditional RAG: retrieve once -> stuff into context -> generate.
# A fixed pipeline, no matter what the question actually needs.
#
# Agentic RAG: an Agents SDK agent with TWO tools - search the FAQ
# knowledge base, or search the live web - where the agent's own
# reasoning decides which to call, whether one search is enough, and
# how many rounds to run before answering. The diagram's "Enough
# information?" loop IS the agent's own tool-calling loop; there's no
# separate check written in this file for it.
# =====================================================================


def traditional_rag(question: str) -> str:
    faqs, similarity = retrieve(question)
    context = "\n\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in faqs)

    response = openai_client.responses.create(
        model=MODEL,
        instructions=(
            "Answer using ONLY the FAQ context below. If it doesn't contain the answer, "
            "say so plainly - do not guess or claim to check anywhere else."
        ),
        input=f"FAQ context (top match similarity {similarity:.0%}):\n{context}\n\nQuestion: {question}",
    )
    return response.output_text


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the internal FAQ knowledge base for a routine support question."""
    faqs, similarity = retrieve(query)
    if similarity < 0.3:
        return "No closely related FAQ entry found."
    return "\n".join(f"- Q: {f['question']} / A: {f['answer']}" for f in faqs)


@function_tool
def search_web(query: str) -> str:
    """Search the live web - use for anything the internal FAQ can't know, e.g. current status/news."""
    results = GoogleSearch({
        "engine": "google",
        "q": query,
        "api_key": os.environ["SERPAPI_API_KEY"],
    }).get_dict()

    organic = results.get("organic_results", [])[:3]
    if not organic:
        return f"No web results found for '{query}'."
    return "\n".join(
        f"- {r.get('title', '')} ({r.get('link', '')}): {r.get('snippet', '')}" for r in organic
    )


agentic_rag_agent = Agent(
    name="AgenticRAGAssistant",
    model=MODEL,
    tools=[search_knowledge_base, search_web],
    instructions="""
Answer the user's question as a customer support assistant.
1. Start with search_knowledge_base - most routine questions are answered there.
2. If the knowledge base doesn't have relevant information, or the question needs
   something the KB can't know (current status, recent news, live conditions),
   call search_web instead.
3. You may call either tool more than once, or both, if one search isn't enough.
4. Once you have enough information, answer directly and briefly. If neither source
   has the answer, say so honestly.
""",
)


def agentic_rag(question: str) -> str:
    result = Runner.run_sync(agentic_rag_agent, question)
    return result.final_output


if __name__ == "__main__":
    questions = [
        "How do I reset my password?",  # covered by the FAQ - both approaches should succeed
        "Is Stripe having any payment outages right now?",  # NOT in the FAQ - needs live web info
        "What's the current price of Bitcoin in USD?",  # NOT in the FAQ - needs live, fast-changing data
    ]

    for question in questions:
        print("=" * 70)
        print(f"Question: {question}\n")

        print("--- Traditional RAG (single retrieval, KB only) ---")
        print(traditional_rag(question))

        print("\n--- Agentic RAG (reasons about which tool(s) to use) ---")
        print(agentic_rag(question))
        print()
