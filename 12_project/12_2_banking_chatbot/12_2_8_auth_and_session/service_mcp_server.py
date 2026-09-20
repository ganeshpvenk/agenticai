import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import mock_bank  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

# =====================================================================
# Service MCP server - Change of Address, Cheque Book Request, KYC
# Update, plus the read-only get_service_details added in step 6. See
# accounts_mcp_server.py's header for the shape this follows.
# =====================================================================

mcp = FastMCP("Service")


@mcp.tool()
def get_service_details(customer_id: str) -> str:
    """Look up a bank customer's current mailing address, cheque book status, and KYC status by their customer ID."""
    result = mock_bank.get_service_details(customer_id)
    if "error" in result:
        return result["error"]
    return (
        f"Service details for {result['name']} (ID {customer_id}): "
        f"Address: {result['address']}. "
        f"Cheque book status: {result['cheque_book_status']}. "
        f"KYC status: {result['kyc_status']}."
    )


@mcp.tool()
def change_of_address(customer_id: str, new_address: str) -> str:
    """Update a bank customer's mailing address by their customer ID."""
    result = mock_bank.update_address(customer_id, new_address)
    if "error" in result:
        return result["error"]
    return f"Address for {result['name']} (ID {customer_id}) updated to: {result['address']}."


@mcp.tool()
def request_cheque_book(customer_id: str) -> str:
    """Request a new cheque book for a bank customer by their customer ID."""
    result = mock_bank.request_cheque_book(customer_id)
    if "error" in result:
        return result["error"]
    return (
        f"Cheque book request for {result['name']} (ID {customer_id}) status: "
        f"{result['cheque_book_status']}."
    )


@mcp.tool()
def update_kyc(customer_id: str, kyc_details: str) -> str:
    """Submit updated KYC details for a bank customer by their customer ID."""
    result = mock_bank.update_kyc(customer_id, kyc_details)
    if "error" in result:
        return result["error"]
    return (
        f"KYC update for {result['name']} (ID {customer_id}) received "
        f"({result['submitted_details']}). Status: {result['kyc_status']}."
    )


if __name__ == "__main__":
    mcp.run()
