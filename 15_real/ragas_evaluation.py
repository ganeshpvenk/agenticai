# pip install ragas openai chromadb sentence-transformers python-dotenv

import os
import sys
import csv
import random
import chromadb
from openai import OpenAI, AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings as RagasHuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecisionWithoutReference
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# RAGAS evaluation of a small retrieve()/generate() RAG pipeline built
# directly on top of the FAQ database (customer_support_qa_500.csv) -
# not on top of an already-built agent, and not against hand-invented
# test questions.
#
# Each test case: take one real (question, answer) pair per category
# from the CSV, use the real answer as the reference, but naturally
# PARAPHRASE the question before asking it - a real user won't type
# the FAQ's exact wording, and testing with the verbatim FAQ question
# would make retrieval trivially easy and prove nothing about the
# pipeline's real-world robustness to phrasing.
#
# Four core RAGAS metrics, each answering a different question:
#   Faithfulness             - does the answer only claim things that
#                               are actually IN the retrieved FAQs, or
#                               did the model add/invent something?
#   Answer Relevancy         - does the answer actually address what
#                               was asked (vs. a vague non-answer)?
#   Context Recall           - did retrieval find what the REAL FAQ
#                               answer needed? (ground truth = the
#                               database's own answer)
#   Context Precision        - of what was retrieved, how much was
#                               actually useful for the answer given?
# =====================================================================

CSV_PATH = os.path.join(os.path.dirname(__file__), "customer_support_qa_500.csv")
MODEL = "gpt-4o-mini"
TOP_K = 3

openai_client = OpenAI()

# ragas's score() calls the LLM's async methods internally, even for "sync" usage - it
# needs an AsyncOpenAI client.
ragas_llm = llm_factory(model=MODEL, provider="openai", client=AsyncOpenAI())
ragas_embeddings = RagasHuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

faithfulness = Faithfulness(llm=ragas_llm)
answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
context_recall = ContextRecall(llm=ragas_llm)
context_precision = ContextPrecisionWithoutReference(llm=ragas_llm)


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
