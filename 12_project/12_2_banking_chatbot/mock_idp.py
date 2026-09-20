"""Mock identity provider for the 12_2_banking_chatbot build (step 8+).

A minimal stand-in for the deck's "Bank IdP / LDAP" (slide 10) - authenticates
a customer_id + PIN pair and returns their auth level. PINs are just the
customer id's digits for demo convenience; this is not meant to model real
credential security, only the shape of "the system knows who is asking."

Two auth levels: "standard" (default) and "restricted". CUST1003 (Sancy) is
deliberately the one restricted account, matching the deck's own example on
slide 10: "John (standard) requests credit limit increase -> Approved; Sancy
(restricted) same request -> Elevated auth required". This build has no
credit-limit tool, so the elevated-auth gate is applied to KYC updates instead
(update_kyc in 12_2_8_auth_and_session.py) - a realistically sensitive,
identity-document-adjacent operation in a real bank.
"""

CREDENTIALS = {
    "CUST1001": {"pin": "1001", "auth_level": "standard"},  # John Mathews
    "CUST1002": {"pin": "1002", "auth_level": "standard"},  # Priya Sharma
    "CUST1003": {"pin": "1003", "auth_level": "restricted"},  # Sancy Fernandes
    "CUST1004": {"pin": "1004", "auth_level": "standard"},  # Sanjay Kulkarni
    "CUST1005": {"pin": "1005", "auth_level": "standard"},  # Ananya Iyer
    "CUST1006": {"pin": "1006", "auth_level": "standard"},  # Vikram Malhotra
}


def authenticate(customer_id: str, pin: str) -> dict:
    record = CREDENTIALS.get(customer_id)
    if record is None or record["pin"] != pin:
        return {"error": "Invalid customer ID or PIN."}
    return {"customer_id": customer_id, "auth_level": record["auth_level"]}
