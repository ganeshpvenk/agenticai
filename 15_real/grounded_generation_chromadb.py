# pip install chromadb openai python-dotenv

import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(__file__), "loan_data")
MODEL = "gpt-4o-mini"
openai_client = OpenAI()

# =====================================================================
# Same idea as grounded_generation_llamaindex.py - answer loan policy
# questions with inline [n] citations to real chunks - but hand-rolled
# with plain OpenAI + ChromaDB instead of LlamaIndex's CitationQueryEngine.
# =====================================================================

collection = chromadb.Client().create_collection("loan_policy_citations")

chunks, ids = [], []
for filename in ["loan_eligibility_policy.md", "rbi_guidelines.md"]:
    text = open(os.path.join(DATA_DIR, filename), encoding="utf-8").read()
    for i, section in enumerate(text.split("\n## ")):
        section = section.strip()
        if section:
            chunks.append(section)
            ids.append(f"{filename}-{i}")

collection.add(ids=ids, documents=chunks)


def ask(question: str):
    results = collection.query(query_texts=[question], n_results=5)
    sources = results["documents"][0]
    numbered_sources = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, start=1))

    response = openai_client.responses.create(
        model=MODEL,
        instructions=(
            "Answer using ONLY the numbered sources below. Cite the source number inline "
            "in square brackets - e.g. [1] - immediately after every claim you make."
        ),
        input=f"SOURCES:\n{numbered_sources}\n\nQuestion: {question}",
    )

    print(f"\nQ: {question}")
    print(f"A: {response.output_text}\n")
    print("Cited sources:")
    for i, s in enumerate(sources, start=1):
        print(f"  [{i}] \"{s[:150].replace(chr(10), ' ')}...\"")


if __name__ == "__main__":
    ask(
        "What is the maximum DTI ratio allowed for a salaried home loan applicant, "
        "and how is the maximum eligible loan amount calculated?"
    )
    ask(
        "What is the maximum loan-to-value ratio for a ₹50 lakh home loan, and what "
        "happens to an application if the applicant's credit score is below 650?"
    )
