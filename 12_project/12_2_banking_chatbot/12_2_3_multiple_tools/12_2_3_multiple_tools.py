import os
import sys
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(override=True)

# mock_bank.py lives one directory up, shared across every step from here on.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import mock_bank  # noqa: E402

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =====================================================================
# Step 3 - Give the agent all six banking tools. Functionally this
# works - even cross-domain requests succeed in one turn - which is
# exactly the problem: one agent now has unrestricted access to every
# domain (balances AND address changes AND KYC updates), with no
# per-domain boundary. That architectural gap is what step 4
# (specialist agents) exists to close - this step does not fix it.
# =====================================================================

SYSTEM_PROMPT = (
    "You are a banking assistant with six tools: get_account_balance, "
    "get_transaction_details, request_statement, change_of_address, "
    "request_cheque_book, and update_kyc. Use whichever tool(s) the "
    "user's request needs - including more than one in the same turn for "
    "cross-domain requests. Never guess a customer ID - if the user doesn't "
    "give one, ask for it. Never invent data that isn't returned by a tool."
)


@tool
def get_account_balance(customer_id: str) -> str:
    """Look up the current account balance for a bank customer by their customer ID."""
    result = mock_bank.get_balance(customer_id)
    if "error" in result:
        return result["error"]
    balance = mock_bank.format_inr(result["balance"])
    return f"Customer {result['name']} (ID {customer_id}) has a balance of {balance}."


@tool
def get_transaction_details(customer_id: str, count: int = 5) -> str:
    """Retrieve a bank customer's most recent transactions by their customer ID."""
    result = mock_bank.get_transactions(customer_id, count)
    if "error" in result:
        return result["error"]
    lines = [
        f"{t['date']}: {t['description']} ({mock_bank.format_inr(t['amount'])})"
        for t in result["transactions"]
    ]
    return f"Recent transactions for {result['name']} (ID {customer_id}):\n" + "\n".join(lines)


@tool
def request_statement(customer_id: str, period: str = "last 30 days") -> str:
    """Generate an account statement for a bank customer for a given period (e.g. 'last 30 days')."""
    result = mock_bank.generate_statement(customer_id, period)
    if "error" in result:
        return result["error"]
    lines = [
        f"{t['date']}: {t['description']} ({mock_bank.format_inr(t['amount'])})"
        for t in result["transactions"]
    ]
    closing = mock_bank.format_inr(result["closing_balance"])
    return (
        f"Statement for {result['name']} (ID {customer_id}), {result['period']}:\n"
        + "\n".join(lines)
        + f"\nClosing balance: {closing}"
    )


@tool
def change_of_address(customer_id: str, new_address: str) -> str:
    """Update a bank customer's mailing address by their customer ID."""
    result = mock_bank.update_address(customer_id, new_address)
    if "error" in result:
        return result["error"]
    return f"Address for {result['name']} (ID {customer_id}) updated to: {result['address']}."


@tool
def request_cheque_book(customer_id: str) -> str:
    """Request a new cheque book for a bank customer by their customer ID."""
    result = mock_bank.request_cheque_book(customer_id)
    if "error" in result:
        return result["error"]
    return (
        f"Cheque book request for {result['name']} (ID {customer_id}) status: "
        f"{result['cheque_book_status']}."
    )


@tool
def update_kyc(customer_id: str, kyc_details: str) -> str:
    """Submit updated KYC details for a bank customer by their customer ID."""
    result = mock_bank.update_kyc(customer_id, kyc_details)
    if "error" in result:
        return result["error"]
    return (
        f"KYC update for {result['name']} (ID {customer_id}) received "
        f"({result['submitted_details']}). Status: {result['kyc_status']}."
    )


tools = [
    get_account_balance,
    get_transaction_details,
    request_statement,
    change_of_address,
    request_cheque_book,
    update_kyc,
]
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

# =====================================================================
# Streamlit web UI. Sidebar buttons fire the deck's demo/regression
# prompts directly - the same ones the console version used to print -
# so this step's README's "demo prompt" / "regression test" sections
# stay testable by clicking rather than reading stdout. Conversation
# state lives in st.session_state for the browser session; there is no
# persistence across restarts yet, that's step 8 (session store).
# =====================================================================

DEMO_PROMPTS = {
    "Cross-domain: balance + transactions": (
        "What is my balance and show me my last 5 transactions for customer CUST1001?"
    ),
    "No domain boundary: address + cheque book": (
        "For customer CUST1003, please update the address to '99 Residency Road, "
        "Bengaluru' and also request a new cheque book."
    ),
    "Regression: balance with ID": "What is the account balance for customer CUST1001?",
    "Regression: cheque book status (no ID)": "What is my cheque book status?",
    "Regression: balance (no ID)": "What is my account balance?",
    "Regression: last 5 transactions (no ID)": "Can you show me my last 5 transactions?",
}

st.set_page_config(page_title="Banking Assistant - Step 3", page_icon="\U0001F3E6")
st.title("Banking Assistant — Step 3: Multiple Tools")
st.caption(
    "One agent, all six banking tools, no domain boundary between them - "
    "that's the problem this step exists to surface, not fix."
)

if "messages" not in st.session_state:
    initial_messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.messages = initial_messages
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    st.header("Demo & regression prompts")
    for label, prompt_text in DEMO_PROMPTS.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt_text
    if st.button("Clear conversation", use_container_width=True):
        cleared_messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        st.session_state.messages = cleared_messages
        st.rerun()

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.write(message.content)

user_input = st.chat_input("Ask about an account...")
prompt_to_run = user_input or st.session_state.pending_prompt

if prompt_to_run:
    st.session_state.pending_prompt = None
    st.session_state.messages.append(HumanMessage(content=prompt_to_run))
    with st.chat_message("user"):
        st.write(prompt_to_run)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": st.session_state.messages})
        st.session_state.messages = result["messages"]
        st.write(result["messages"][-1].content)
