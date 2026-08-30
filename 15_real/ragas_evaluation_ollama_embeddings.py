# pip install ragas openai chromadb ollama python-dotenv
# Requires a local Ollama server with the embedding model pulled:
#   ollama pull nomic-embed-text

import os
import sys
import csv
import random
import time
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from openai import OpenAI, AsyncOpenAI
import ollama
from ollama import ResponseError
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecisionWithoutReference
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Ollama embeddings variant of ragas_evaluation.py: both the retrieval
# index and the RAGAS answer-relevancy metric embed text locally via a
# running Ollama server (nomic-embed-text) instead of downloading a
# HuggingFace sentence-transformers model. Everything else - the RAG
# pipeline built directly on the FAQ database, the paraphrase-then-ask
# test cases, the four RAGAS metrics - is unchanged from that file.
# =====================================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")
MODEL = "gpt-4o-mini"
TOP_K = 3

openai_client = OpenAI()

# ragas's score() calls the LLM's async methods internally, even for "sync" usage - it
# needs an AsyncOpenAI client.
ragas_llm = llm_factory(model=MODEL, provider="openai", client=AsyncOpenAI())


def _embed_with_retry(embed_call):
    """Ollama's server proxies each embed request to a per-model runner
    subprocess, and occasionally does so before that subprocess has
    finished starting - a transient server-side race that surfaces as a
    connection-refused error on an internal port, not a bad request.
    Retry a couple of times before giving up."""
    for attempt in range(3):
        try:
            return embed_call()
        except ResponseError:
            if attempt == 2:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


async def _aembed_with_retry(embed_call):
    for attempt in range(3):
        try:
            return await embed_call()
        except ResponseError:
            if attempt == 2:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


class OllamaRagasEmbedding(BaseRagasEmbedding):
    """ragas.metrics.collections (the modern metrics API used below) requires a
    BaseRagasEmbedding, and ragas ships no built-in Ollama provider - so call
    the Ollama embed endpoint directly.

    aembed_text creates a fresh ollama.AsyncClient() per call rather than
    reusing one: each score() call wraps its work in its own asyncio.run(),
    and an async client's connection pool stays bound to the event loop it
    was first used on - reusing it across score() calls fails with
    "Event loop is closed" once that first loop ends.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        super().__init__()
        self._model = model
        self._sync_client = ollama.Client()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        response = _embed_with_retry(lambda: self._sync_client.embed(model=self._model, input=text))
        return list(response.embeddings[0])

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        response = await _aembed_with_retry(lambda: ollama.AsyncClient().embed(model=self._model, input=text))
        return list(response.embeddings[0])


ragas_embeddings = OllamaRagasEmbedding(model="nomic-embed-text")

faithfulness = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_recall = ContextRecall(llm=ragas_llm)
context_precision = ContextPrecisionWithoutReference(llm=ragas_llm)


# ---- Minimal RAG pipeline, built directly on the FAQ database ----


class RetryingOllamaEmbeddingFunction(OllamaEmbeddingFunction):
    """See _embed_with_retry above - chromadb's Ollama embedding function
    hits the same transient runner-startup race, so retry it too."""

    def __call__(self, input):
        return _embed_with_retry(lambda: super(RetryingOllamaEmbeddingFunction, self).__call__(input))


with open(CSV_PATH, newline="", encoding="utf-8") as f:
    FAQ_ROWS = list(csv.DictReader(f))

FAQ_INDEX = chromadb.Client().create_collection(
    "support_faq",
    metadata={"hnsw:space": "cosine"},
    embedding_function=RetryingOllamaEmbeddingFunction(model_name="nomic-embed-text"),
)
FAQ_INDEX.add(
    ids=[row["id"] for row in FAQ_ROWS],
    documents=[row["question"] for row in FAQ_ROWS],
    metadatas=[{"answer": row["answer"]} for row in FAQ_ROWS],
)


def retrieve(question: str) -> list[dict]:
    result = FAQ_INDEX.query(query_texts=[question], n_results=TOP_K)
    return [
        {"question": q, "answer": meta["answer"]}
        for q, meta in zip(result["documents"][0], result["metadatas"][0])
    ]


def generate(question: str, faqs: list[dict]) -> str:
    context = "\n\n".join(f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs)
    response = openai_client.responses.create(
        model=MODEL,
        instructions=(
            "You are a customer support assistant. Answer using ONLY the FAQ context below - "
            "do not invent policies. Keep it short. If the context doesn't answer the "
            "question, say a support agent will need to follow up."
        ),
        input=f"FAQ context:\n{context}\n\nCustomer message: {question}",
    )
    return response.output_text


def paraphrase(question: str) -> str:
    """Rephrase a real FAQ question the way an actual user would type it - same
    meaning, different wording - so retrieval is tested on realistic input."""
    return openai_client.responses.create(
        model=MODEL,
        instructions="Rephrase this customer support question naturally and casually, the "
                     "way a real user typing quickly would. Keep the same meaning. Return "
                     "ONLY the rephrased question, nothing else.",
        input=question,
    ).output_text.strip()


def build_test_cases(seed: int = 42) -> list[dict]:
    """Sample one real (question, answer) pair per category from the FAQ database
    as the eval set, then paraphrase each question before it's asked."""
    by_category: dict[str, list[dict]] = {}
    for row in FAQ_ROWS:
        by_category.setdefault(row["category"], []).append(row)

    random.seed(seed)
    return [
        {
            "category": category,
            "original_question": entry["question"],
            "question": paraphrase(entry["question"]),
            "reference": entry["answer"],
        }
        for category, entries in sorted(by_category.items())
        for entry in [random.choice(entries)]
    ]


def evaluate_case(question: str, reference: str) -> dict:
    faqs = retrieve(question)
    retrieved_contexts = [f"Q: {faq['question']} A: {faq['answer']}" for faq in faqs]
    response = generate(question, faqs)

    return {
        "question": question,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "faithfulness": faithfulness.score(
            user_input=question, response=response, retrieved_contexts=retrieved_contexts
        ).value,
        "answer_relevancy": answer_relevancy.score(user_input=question, response=response).value,
        "context_recall": context_recall.score(
            user_input=question, retrieved_contexts=retrieved_contexts, reference=reference
        ).value,
        "context_precision": context_precision.score(
            user_input=question, response=response, retrieved_contexts=retrieved_contexts
        ).value,
    }


if __name__ == "__main__":
    results = []

    for case in build_test_cases():
        print("=" * 70)
        print(f"Category: {case['category']}")
        print(f"Original FAQ question: {case['original_question']}")
        print(f"Paraphrased as asked:  {case['question']}")

        result = evaluate_case(case["question"], case["reference"])
        results.append(result)

        print(f"Response: {result['response']}")
        print(f"Faithfulness:       {result['faithfulness']:.2f}")
        print(f"Answer Relevancy:   {result['answer_relevancy']:.2f}")
        print(f"Context Recall:     {result['context_recall']:.2f}")
        print(f"Context Precision:  {result['context_precision']:.2f}")
        print()

    print("=" * 70)
    print("AVERAGES ACROSS ALL TEST CASES")
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"  {metric}: {avg:.2f}")
