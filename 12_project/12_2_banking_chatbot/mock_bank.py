"""Shared SQLite-backed mock bank database for the 12_2_banking_chatbot build.

Every step subdirectory from 12_2_2_single_tool onward imports this module
(via BASE_DIR/.. on sys.path) rather than redefining customer data. Started as
an in-memory dict; step 2/3 needed only balances then the other five domains,
and now the whole thing lives in bank.db (SQLite, stdlib sqlite3 - no new
dependency) so state actually persists across runs, the way a real banking
backend would. The public functions below (get_balance, get_transactions,
etc.) keep the exact same names/signatures/return shapes they had as an
in-memory dict, so no calling code in step 2 or step 3 changed at all.

The deck this build follows (Banking-Agentic-AI.pptx) is an Indian retail
bank (RBI/DPDP Act references, "4.2 Lakh calls/month", rupee figures like
"₹1,16,200"), so balances are in INR using Indian digit grouping, and three
customer ids (CUST1001/1003/1004) are reserved to match named characters the
deck itself uses in later steps: John (slide 10, standard auth - approved),
Sancy (slide 10, restricted auth - elevated auth required), and Sanjay
(slide 11, the customer who flagged a suspicious transaction last week - see
CUST1004's most recent transaction below, flagged for exactly that reason).

Because state now persists in bank.db, demo mutations (address changes,
cheque-book requests, KYC updates) stick around across separate runs instead
of resetting each time. Call reset_db() to wipe bank.db and reseed it back to
the values below.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.db")

_SEED_CUSTOMERS = {
    "CUST1001": {
        "name": "John Mathews",
        "balance": 52340.75,
        "address": "12 MG Road, Bengaluru",
        "cheque_book_status": "Not requested",
        "kyc_status": "Verified",
        "transactions": [
            {"date": "2026-09-01", "description": "UPI - Swiggy", "amount": -450.00},
            {"date": "2026-08-29", "description": "Salary Credit", "amount": 85000.00},
            {"date": "2026-08-25", "description": "ATM Withdrawal", "amount": -5000.00},
            {"date": "2026-08-20", "description": "Electricity Bill - BESCOM", "amount": -1875.50},
            {"date": "2026-08-15", "description": "Insurance Premium - LIC", "amount": -6200.00},
        ],
    },
    "CUST1002": {
        "name": "Priya Sharma",
        "balance": 118750.00,
        "address": "45 Park Street, Kolkata",
        "cheque_book_status": "Not requested",
        "kyc_status": "Verified",
        "transactions": [
            {"date": "2026-09-02", "description": "UPI - Amazon", "amount": -2340.00},
            {"date": "2026-08-30", "description": "Salary Credit", "amount": 95000.00},
            {"date": "2026-08-22", "description": "Mutual Fund SIP", "amount": -10000.00},
            {"date": "2026-08-15", "description": "Credit Card Bill Payment", "amount": -18500.00},
        ],
    },
    "CUST1003": {
        "name": "Sancy Fernandes",
        "balance": 8420.50,
        "address": "7 Linking Road, Mumbai",
        "cheque_book_status": "Dispatched",
        "kyc_status": "Verified",
        "transactions": [
            {"date": "2026-09-03", "description": "UPI - Zomato", "amount": -620.00},
            {"date": "2026-08-28", "description": "Freelance Payment Credit", "amount": 15000.00},
            {"date": "2026-08-18", "description": "ATM Withdrawal", "amount": -2000.00},
            {"date": "2026-08-10", "description": "Mobile Recharge", "amount": -399.00},
        ],
    },
    "CUST1004": {
        "name": "Sanjay Kulkarni",
        "balance": 275600.00,
        "address": "18 FC Road, Pune",
        "cheque_book_status": "Not requested",
        "kyc_status": "Pending review",
        "transactions": [
            {"date": "2026-09-04", "description": "UPI - Unknown Merchant (flagged)", "amount": -48000.00},
            {"date": "2026-08-31", "description": "Salary Credit", "amount": 120000.00},
            {"date": "2026-08-19", "description": "Home Loan EMI", "amount": -32000.00},
            {"date": "2026-08-12", "description": "UPI - Grocery Store", "amount": -1560.00},
        ],
    },
    "CUST1005": {
        "name": "Ananya Iyer",
        "balance": 963.25,
        "address": "3 Anna Salai, Chennai",
        "cheque_book_status": "Not requested",
        "kyc_status": "Verified",
        "transactions": [
            {"date": "2026-09-01", "description": "UPI - College Fees", "amount": -12000.00},
            {"date": "2026-08-26", "description": "Part-time Stipend Credit", "amount": 6000.00},
            {"date": "2026-08-14", "description": "ATM Withdrawal", "amount": -1000.00},
            {"date": "2026-08-05", "description": "UPI - Bookstore", "amount": -540.00},
        ],
    },
    "CUST1006": {
        "name": "Vikram Malhotra",
        "balance": 450000.00,
        "address": "56 Sector 17, Chandigarh",
        "cheque_book_status": "Requested - processing",
        "kyc_status": "Verified",
        "transactions": [
            {"date": "2026-08-30", "description": "Salary Credit", "amount": 210000.00},
            {"date": "2026-08-27", "description": "Stock Broker Transfer", "amount": -50000.00},
            {"date": "2026-08-21", "description": "Property Tax Payment", "amount": -25000.00},
            {"date": "2026-08-11", "description": "UPI - Restaurant", "amount": -3200.00},
        ],
    },
}


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            balance REAL NOT NULL,
            address TEXT NOT NULL,
            cheque_book_status TEXT NOT NULL,
            kyc_status TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL REFERENCES customers(customer_id),
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL
        )"""
    )
    for customer_id, data in _SEED_CUSTOMERS.items():
        conn.execute(
            "INSERT INTO customers (customer_id, name, balance, address, cheque_book_status, kyc_status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                customer_id,
                data["name"],
                data["balance"],
                data["address"],
                data["cheque_book_status"],
                data["kyc_status"],
            ),
        )
        for t in data["transactions"]:
            conn.execute(
                "INSERT INTO transactions (customer_id, date, description, amount) VALUES (?, ?, ?, ?)",
                (customer_id, t["date"], t["description"], t["amount"]),
            )
    conn.commit()


def reset_db() -> None:
    """Delete bank.db and reseed it - restores every customer to the values above."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = _get_connection()
    try:
        _seed(conn)
    finally:
        conn.close()


if not os.path.exists(DB_PATH):
    _conn = _get_connection()
    try:
        _seed(_conn)
    finally:
        _conn.close()


def format_inr(amount: float) -> str:
    """Format a number using the Indian digit-grouping convention (lakhs/crores),
    e.g. 118750.0 -> '₹1,18,750.00', matching the deck's own '₹1,16,200' style."""
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    if len(whole) <= 3:
        grouped = whole
    else:
        last3, rest = whole[-3:], whole[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups) + "," + last3
    return f"{sign}₹{grouped}.{frac}"


def get_balance(customer_id: str) -> dict:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT name, balance FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": f"No customer found with id {customer_id}"}
    return {"customer_id": customer_id, "name": row["name"], "balance": row["balance"]}


def get_transactions(customer_id: str, count: int = 5) -> dict:
    conn = _get_connection()
    try:
        customer = conn.execute(
            "SELECT name FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if customer is None:
            return {"error": f"No customer found with id {customer_id}"}
        rows = conn.execute(
            "SELECT date, description, amount FROM transactions "
            "WHERE customer_id = ? ORDER BY date DESC, id DESC LIMIT ?",
            (customer_id, count),
        ).fetchall()
    finally:
        conn.close()
    return {
        "customer_id": customer_id,
        "name": customer["name"],
        "transactions": [dict(r) for r in rows],
    }


def generate_statement(customer_id: str, period: str = "last 30 days") -> dict:
    conn = _get_connection()
    try:
        customer = conn.execute(
            "SELECT name, balance FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()
        if customer is None:
            return {"error": f"No customer found with id {customer_id}"}
        rows = conn.execute(
            "SELECT date, description, amount FROM transactions "
            "WHERE customer_id = ? ORDER BY date DESC, id DESC",
            (customer_id,),
        ).fetchall()
    finally:
        conn.close()
    return {
        "customer_id": customer_id,
        "name": customer["name"],
        "period": period,
        "transactions": [dict(r) for r in rows],
        "closing_balance": customer["balance"],
    }


def get_service_details(customer_id: str) -> dict:
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT name, address, cheque_book_status, kyc_status FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": f"No customer found with id {customer_id}"}
    return {
        "customer_id": customer_id,
        "name": row["name"],
        "address": row["address"],
        "cheque_book_status": row["cheque_book_status"],
        "kyc_status": row["kyc_status"],
    }


def update_address(customer_id: str, new_address: str) -> dict:
    conn = _get_connection()
    try:
        cur = conn.execute(
            "UPDATE customers SET address = ? WHERE customer_id = ?", (new_address, customer_id)
        )
        if cur.rowcount == 0:
            return {"error": f"No customer found with id {customer_id}"}
        conn.commit()
        name = conn.execute(
            "SELECT name FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()["name"]
    finally:
        conn.close()
    return {"customer_id": customer_id, "name": name, "address": new_address}


def request_cheque_book(customer_id: str) -> dict:
    conn = _get_connection()
    try:
        status = "Requested - processing"
        cur = conn.execute(
            "UPDATE customers SET cheque_book_status = ? WHERE customer_id = ?", (status, customer_id)
        )
        if cur.rowcount == 0:
            return {"error": f"No customer found with id {customer_id}"}
        conn.commit()
        name = conn.execute(
            "SELECT name FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()["name"]
    finally:
        conn.close()
    return {"customer_id": customer_id, "name": name, "cheque_book_status": status}


def update_kyc(customer_id: str, kyc_details: str) -> dict:
    conn = _get_connection()
    try:
        status = "Pending review"
        cur = conn.execute(
            "UPDATE customers SET kyc_status = ? WHERE customer_id = ?", (status, customer_id)
        )
        if cur.rowcount == 0:
            return {"error": f"No customer found with id {customer_id}"}
        conn.commit()
        name = conn.execute(
            "SELECT name FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()["name"]
    finally:
        conn.close()
    return {
        "customer_id": customer_id,
        "name": name,
        "kyc_status": status,
        "submitted_details": kyc_details,
    }
