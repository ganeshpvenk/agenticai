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
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# HyDE (Hypothetical Document Embeddings): a short, informally-phrased
# question often sits far away in embedding space from the formal,
# jargon-heavy passage that actually answers it - the two just don't use
# the same words. HyDE closes that gap by asking an LLM to draft a fake
# ANSWER to the question first (it doesn't have to be correct - it just
# has to sound like the corpus), then embeds and searches with THAT
# hypothetical document instead of the raw question. Retrieval becomes a
# document-to-document match instead of a question-to-document match.
#
# Knowledge base: 50 insurance policy statements (health, motor, life,
# travel, home). The two questions below are deliberately phrased the
# way a customer would actually ask them - casual, no policy jargon -
# to widen the gap HyDE has to close against the formal wording of the
# real clauses.
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "hyde_chroma_db")
shutil.rmtree(PERSIST_DIR, ignore_errors=True)

with open(os.path.join(BASE_DIR, "policy_documents.json"), encoding="utf-8") as f:
    POLICY_DOCUMENTS = json.load(f)

vectorstore = Chroma(
    collection_name="policy_docs",
    embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory=PERSIST_DIR,
)
vectorstore.add_texts(POLICY_DOCUMENTS)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

QUESTIONS = [
    "If I get admitted to a hospital that's in my insurer's network, do I need to pay out of "
    "my own pocket first, or can the bills go straight to the insurance company?",

    "I just bought health cover last month - if something happens with a health issue I "
    "already had before buying it, will insurance actually pay for it right away?",
]


def show(label: str, results) -> None:
    print(f"\n{label}")
    for doc, score in results:
        print(f"  distance={score:.4f}  {doc.page_content}")


for question in QUESTIONS:
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # -----------------------------
    # BEFORE: embed the question as-is. Chroma's score here is a
    # distance - lower means more similar.
    # -----------------------------
    direct_results = vectorstore.similarity_search_with_score(question, k=3)
    show("BEFORE (direct question embedding):", direct_results)

    # -----------------------------
    # HyDE: draft a hypothetical passage that WOULD answer the question,
    # in the same formal policy-document style as the corpus - then
    # embed and search with that instead.
    # -----------------------------
    hyde_prompt = (
        "Write one short, confident sentence of insurance policy documentation that would "
        f"answer this question, in the style of a formal policy wording document. Do not "
        f"hedge or mention that it's hypothetical.\n\nQuestion: {question}"
    )
    hypothetical_doc = llm.invoke(hyde_prompt).content
    print(f"\nHYPOTHETICAL DOCUMENT: {hypothetical_doc}")

    hyde_results = vectorstore.similarity_search_with_score(hypothetical_doc, k=3)
    show("AFTER (HyDE - embedding the hypothetical document):", hyde_results)
