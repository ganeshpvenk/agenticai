import os
import sys
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(override=True)

# Tool/LLM output can contain characters outside Windows' default console
# codepage (cp1252) - reconfigure stdout to UTF-8 so printing doesn't crash.
sys.stdout.reconfigure(encoding="utf-8")

# mock_bank.py lives one directory up, shared across every step from here on.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import mock_bank  # noqa: E402

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================================
# Step 2 - Give the agent one real tool: Balance Enquiry, wired to the
# shared mock bank backend. Standard LangGraph tool-calling loop:
# agent decides whether to call the tool, ToolNode executes it, the
# result is fed back to the agent for a natural-language answer.
# =====================================================================

SYSTEM_PROMPT = (
    "You are a banking assistant. You have exactly one tool available: "
    "get_account_balance, which looks up a customer's balance by customer ID. "
    "You have no other banking capability yet - if asked about anything else "
    "(transactions, statements, address, cheque book, KYC), clearly say you "
    "don't have that capability yet. Never guess a customer ID - if the user "
    "doesn't give one, ask for it. Never invent a balance or any other figure."
)


@tool
def get_account_balance(customer_id: str) -> str:
    """Look up the current account balance for a bank customer by their customer ID."""
    result = mock_bank.get_balance(customer_id)
    if "error" in result:
        return result["error"]
    balance = mock_bank.format_inr(result["balance"])
    return f"Customer {result['name']} (ID {customer_id}) has a balance of {balance}."


tools = [get_account_balance]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def agent(state: ChatState) -> ChatState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: ChatState) -> str:
    return "tools" if state["messages"][-1].tool_calls else "end"


graph = StateGraph(ChatState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")

app = graph.compile()

if __name__ == "__main__":
    demo_prompts = [
        "What is the account balance for customer CUST1001?",
        "What is my cheque book status?",
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
