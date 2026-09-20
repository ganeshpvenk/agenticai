import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv(override=True)

# Tool/LLM output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================================
# Step 1 - Naive agent: one LangGraph node, no tools, no bank connection.
# START -> generate -> END. The agent can only answer from LLM training
# knowledge, so any banking question that needs live data must fail
# honestly rather than invent an answer.
# =====================================================================

SYSTEM_PROMPT = (
    "You are a banking assistant. You currently have no connection to any "
    "bank system - no tools, no account data. If asked about account "
    "balances, transactions, or any customer-specific banking information, "
    "clearly say you don't have access to that data. Never invent a figure."
)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def generate(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("generate", generate)
graph.add_edge(START, "generate")
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    demo_prompts = [
        "What is my account balance?",
        "Can you show me my last 5 transactions?",
    ]

    for prompt in demo_prompts:
        print("=" * 70)
        print(f"User: {prompt}")
        result = app.invoke(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            }
        )
        print(f"Assistant: {result['messages'][-1].content}")
