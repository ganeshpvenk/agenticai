# pip install deepeval openai chromadb python-dotenv
# Requires a local Ollama server with the embedding model pulled:
#   ollama pull nomic-embed-text

import os
import sys
import csv
import random
import time
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from ollama import ResponseError
from openai import OpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ToxicityMetric,
    BiasMetric,
)
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Ollama embeddings variant of deepeval_evaluation.py: the FAQ retrieval
# index embeds text locally via a running Ollama server (nomic-embed-text)
# instead of downloading a HuggingFace sentence-transformers model.
#
# Unlike the RAGAS AnswerRelevancy metric, none of DeepEval's metrics
# here take an embeddings argument - they're all LLM-judge based (each
# metric just gets `model=MODEL`), so there's no second embeddings
# object to swap. Only the retrieval index changes.
#
# Toxicity and Bias are content-safety metrics with no RAGAS equivalent
# in the ragas_evaluation*.py comparison - they score the generation
# itself, not its groundedness in the retrieved FAQs.
# =====================================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")
MODEL = "gpt-4o-mini"
TOP_K = 3

openai_client = OpenAI()

faithfulness = FaithfulnessMetric(model=MODEL)
answer_relevancy = AnswerRelevancyMetric(model=MODEL)
contextual_recall = ContextualRecallMetric(model=MODEL)
contextual_precision = ContextualPrecisionMetric(model=MODEL)
toxicity = ToxicityMetric(model=MODEL)
bias = BiasMetric(model=MODEL)


# ---- Minimal RAG pipeline, built directly on the FAQ database ----


class RetryingOllamaEmbeddingFunction(OllamaEmbeddingFunction):
    """Ollama's server proxies each embed request to a per-model runner
    subprocess, and occasionally does so before that subprocess has
    finished starting - a transient server-side race that surfaces as a
    connection-refused error on an internal port, not a bad request.
    Retry a couple of times before giving up."""

    def __call__(self, input):
        for attempt in range(3):
            try:
                return super().__call__(input)
            except ResponseError:
                if attempt == 2:
                    raise
                time.sleep(2)
        raise AssertionError("unreachable")


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

    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        expected_output=reference,
        retrieval_context=retrieved_contexts,
    )

    faithfulness.measure(test_case)
    answer_relevancy.measure(test_case)
    contextual_recall.measure(test_case)
    contextual_precision.measure(test_case)
    toxicity.measure(test_case)
    bias.measure(test_case)

    return {
        "question": question,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "faithfulness": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
        "answer_relevancy": answer_relevancy.score,
        "answer_relevancy_reason": answer_relevancy.reason,
        "contextual_recall": contextual_recall.score,
        "contextual_precision": contextual_precision.score,
        "toxicity": toxicity.score,
        "toxicity_reason": toxicity.reason,
        "bias": bias.score,
        "bias_reason": bias.reason,
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
        print(f"Faithfulness:         {result['faithfulness']:.2f}  ({result['faithfulness_reason']})")
        print(f"Answer Relevancy:     {result['answer_relevancy']:.2f}  ({result['answer_relevancy_reason']})")
        print(f"Contextual Recall:    {result['contextual_recall']:.2f}")
        print(f"Contextual Precision: {result['contextual_precision']:.2f}")
        print(f"Toxicity:             {result['toxicity']:.2f}  ({result['toxicity_reason']})")
        print(f"Bias:                 {result['bias']:.2f}  ({result['bias_reason']})")
        print()

    print("=" * 70)
    print("AVERAGES ACROSS ALL TEST CASES")
    for metric in ["faithfulness", "answer_relevancy", "contextual_recall", "contextual_precision", "toxicity", "bias"]:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"  {metric}: {avg:.2f}")
