# pip install llama-cloud-services python-dotenv
# Needs a LLAMA_CLOUD_API_KEY from https://cloud.llamaindex.ai (free tier available).

import os
import sys
from llama_cloud_services import LlamaParse
from dotenv import load_dotenv

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "ocr.png")

# =====================================================================
# ocr.png is a scanned Spanish government employment contract form -
# dense, boxed fields, checkboxes and a two-column layout. A plain
# vision-model OCR call (see 14_advanced/05_langchain/telugu_ocr.py)
# just transcribes characters; LlamaParse is a document-parsing service
# built to additionally recover the document's *structure* - it renders
# the form as clean markdown with headings, tables and checkbox state
# preserved, which is what you actually want if the extracted text is
# going to be chunked/indexed for RAG afterwards.
# =====================================================================

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    language="es",  # form is in Spanish
    verbose=True,
)

documents = parser.load_data(IMAGE_PATH)

for i, doc in enumerate(documents, start=1):
    print(f"\n--- Page {i} ---")
    print(doc.text)
