import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import json
import os
import shutil
import sys
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# Multi-Query Retrieval: a single query is just one way of phrasing a
# need, but the document that actually answers it might use completely
# different wording. Relying on one embedding means relying on one
# phrasing matching the source text.
#
# Fix: instead of searching with one query, ask an LLM to generate
# several alternative reformulations of it, search with EACH of them
# separately, then merge and deduplicate the results into one combined
# candidate pool. This trades recall for redundant work - more retrieval
# calls, and the merged pool needs deduplication since the same document
# can easily surface from more than one reformulation. It's typically
# paired with a reranker afterward (see cohere_reranking.py in this
# folder) to cut the larger combined pool back down to the few that
# actually matter.
#
# Knowledge base: the same 50-statement insurance policy KB used
# elsewhere in this folder. The two questions below are phrased the
# vague, keyword-poor way a real customer would type them - exactly the
# case where a single embedding search misses documents that use more
# precise policy terminology.
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "multiquery_chroma_db")
shutil.rmtree(PERSIST_DIR, ignore_errors=True)

with open(os.path.join(BASE_DIR, "policy_documents.json"), encoding="utf-8") as f:
    POLICY_DOCUMENTS = json.load(f)

vectorstore = Chroma(
    collection_name="policy_docs",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory=PERSIST_DIR,
)
vectorstore.add_texts(POLICY_DOCUMENTS)

reformulate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class Reformulations(BaseModel):
    queries: list[str] = Field(
        description="3 to 4 alternative phrasings of the question, each emphasizing "
        "different keywords or angles that a relevant document might actually use."
    )


QUESTIONS = [
    "Why did my hospital claim get rejected even though I have insurance?",
    "How do I get my money back for medicines and tests after I already paid the hospital myself?",
]


def retrieve(query: str, k: int = 4) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


for question in QUESTIONS:
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # -----------------------------
    # BEFORE: a single query, embedded and searched once.
    # -----------------------------
    before_docs = retrieve(question)
    print("\nBEFORE (single-query retrieval):")
    for i, doc in enumerate(before_docs, start=1):
        print(f"  [{i}] {doc.page_content}")

    # -----------------------------
    # Generate alternative reformulations of the same underlying need.
    # -----------------------------
    reformulations = reformulate_llm.with_structured_output(Reformulations).invoke(
        "Generate 3 to 4 alternative phrasings of this question that a person might use "
        "when searching an insurance knowledge base. Vary the keywords and angle - e.g. "
        "one version closer to formal policy terminology, one focused on documents/process, "
        f"one focused on timing or eligibility.\n\nQuestion: {question}"
    ).queries

    print("\nLLM-GENERATED REFORMULATIONS:")
    for q in reformulations:
        print(f"  - {q}")

    # -----------------------------
    # AFTER: search with the original query AND every reformulation,
    # then merge results and deduplicate by document content (the same
    # document commonly surfaces from more than one reformulation).
    # -----------------------------
    merged: dict[str, Document] = {}
    for q in [question] + reformulations:
        for doc in retrieve(q):
            merged.setdefault(doc.page_content, doc)
    merged_docs = list(merged.values())

    print(f"\nAFTER (multi-query retrieval, merged + deduplicated -> {len(merged_docs)} candidates):")
    for i, doc in enumerate(merged_docs, start=1):
        print(f"  [{i}] {doc.page_content}")

    # -----------------------------
    # Payoff: which documents did multi-query surface that the single
    # original query alone would have missed entirely?
    # -----------------------------
    before_texts = {doc.page_content for doc in before_docs}
    new_docs = [doc for doc in merged_docs if doc.page_content not in before_texts]

    print(f"\n{len(new_docs)} additional document(s) surfaced only via reformulations:")
    for doc in new_docs:
        print(f"  + {doc.page_content}")
