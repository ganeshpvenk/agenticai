import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import mock_bank  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

# =====================================================================
# Transaction MCP server - Transaction Details + Statement Request,
# moved out of the agent process. See accounts_mcp_server.py's header
# for the shape this follows.
# =====================================================================

mcp = FastMCP("Transaction")


@mcp.tool()
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


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run()
