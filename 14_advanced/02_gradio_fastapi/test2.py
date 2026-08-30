from fastapi import FastAPI
from openai import AsyncOpenAI
import asyncio

app = FastAPI()
client = AsyncOpenAI()

async def ask_llm(prompt: str):
    response = await client.responses.create(
        model="gpt-5",
        input=prompt
    )
    return response.output_text


@app.post("/analyze")
async def analyze(text: str):

    # Launch all LLM calls concurrently
    summary_task = ask_llm(f"Summarize:\n{text}")
    keywords_task = ask_llm(f"Extract five keywords:\n{text}")
    sentiment_task = ask_llm(f"Determine the sentiment:\n{text}")

    # Wait for all three to complete
    summary, keywords, sentiment = await asyncio.gather(
        summary_task,
        keywords_task,
        sentiment_task
    )

    return {
        "summary": summary,
        "keywords": keywords,
        "sentiment": sentiment
    }
