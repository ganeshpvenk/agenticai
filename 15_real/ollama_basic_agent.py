# pip install openai
#
# Requires a local Ollama server running the qwen3 model:
#   ollama pull qwen3:8b
#   ollama serve   (usually already running as a background service)

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain agentic AI in one paragraph"}
    ]
)

print(response.choices[0].message.content)
