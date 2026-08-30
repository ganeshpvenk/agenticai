from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    #instructions="You are a person who understands AI very well.",
    instructions="You are a seasoned political analyst with a deep understanding of global affairs.",
    #input="Explain Artificial Intelligence.",
    input="What is your opinion on Donald Trump as a Leader? Do you approve his third term as US President?",
    temperature=0.9
)

print(response.output_text)
