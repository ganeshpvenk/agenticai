# pip install deepeval openai chromadb python-dotenv

import os
import sys
import csv
import random
import chromadb
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
# DeepEval evaluation of a small retrieve()/generate() RAG pipeline
# built directly on top of the FAQ database (customer_support_qa_500.csv)
# - not on top of an already-built agent - on the same real, database-
# sourced test cases as 15_real/ragas_evaluation.py (see that file for
# the full rationale: real FAQ ground truth + paraphrased questions,
# not hand-invented test cases).
#
# This file exists to contrast frameworks, not pipelines: RAGAS scores
# metrics itself via its own `.score()` calls; DeepEval instead hands
# an LLMTestCase to each metric's `.measure()` and reads `.score`/
# `.reason` back - one added benefit being a human-readable reason
# string per metric, not just a number.
#
# Same four RAG-quality metrics, same meaning as the RAGAS example:
#   Faithfulness             - does the answer only claim things that
#                               are actually IN the retrieved FAQs, or
#                               did the model add/invent something?
#   Answer Relevancy         - does the answer actually address what
#                               was asked (vs. a vague non-answer)?
#   Contextual Recall        - did retrieval find what the REAL FAQ
#                               answer needed? (expected_output = the
#                               database's own answer)
#   Contextual Precision     - of what was retrieved, how much was
#                               actually useful for the answer given?
#
# Plus two content-safety metrics with no RAGAS equivalent in this
# comparison - they score the generation itself, not its groundedness:
#   Toxicity                 - does the answer contain toxic language?
#   Bias                     - does the answer show gender/racial/
#                               political/etc. bias?
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

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    FAQ_ROWS = list(csv.DictReader(f))

FAQ_INDEX = chromadb.Client().create_collection("support_faq", metadata={"hnsw:space": "cosine"})
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
