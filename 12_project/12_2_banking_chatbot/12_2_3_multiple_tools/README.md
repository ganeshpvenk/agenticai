# Step 3 — Multiple Tools: Six Banking Tools, One Unscoped Agent

## Files changed

- `../mock_bank.py` — extended with transactions, address, cheque-book status and
  KYC status per customer, plus `get_transactions`, `generate_statement`,
  `update_address`, `request_cheque_book`, `update_kyc`. `get_balance` and
  `format_inr` from step 2 are unchanged (only `format_inr`'s negative-sign
  placement was tightened: `-₹450.00` instead of `₹-450.00`). Also migrated from
  an in-memory dict to SQLite (`bank.db`, two tables: `customers`, `transactions`)
  — same function signatures/return shapes, so this file's tool wrappers didn't
  need to change for the swap. Because state now really persists, the "no domain
  boundary" demo below writes a change to `bank.db` that outlives the run — run
  `python -c "import mock_bank; mock_bank.reset_db()"` from `../` to reseed.
- `12_2_3_multiple_tools.py` (new) — same agent/tools graph as step 2, now with
  all six tools bound to the one agent, behind a **Streamlit chat UI** instead of
  a console script. Streamlit is the UI for every step from here on — added to
  root `requirements.txt` (`streamlit==1.63.0`) and installed into `.venv`.

## Architecture change

Same loop as step 2 (`agent ⇄ tools`), just six tools instead of one:
`get_account_balance`, `get_transaction_details`, `request_statement`,
`change_of_address`, `request_cheque_book`, `update_kyc`. No new agent
architecture — that's the point of this step. The UI layer did change: a
Streamlit chat window with a sidebar of one-click demo/regression prompts,
replacing the console `print()` runner. Conversation state lives in
`st.session_state` for the browser session only — no persistence across
restarts yet, that's step 8 (session store).

## Demo prompt

Click **"Cross-domain: balance + transactions"** in the sidebar, or type into
the chat box:

```
What is my balance and show me my last 5 transactions for customer CUST1001?
```

## Expected output

The agent calls both `get_account_balance` and `get_transaction_details` in the
same turn and merges the results into one answer — a cross-domain request now
succeeds, which it couldn't in step 2.

## Failure case — this step's real point

This step doesn't fail functionally. It succeeds at everything, and *that's* the
problem the deck deliberately surfaces here. Click **"No domain boundary: address
+ cheque book"** in the sidebar, or type:

```
For customer CUST1003, please update the address to '99 Residency Road,
Bengaluru' and also request a new cheque book.
```

The agent does both without hesitation — an address change (Service domain) and a
cheque-book request (also Service, but a materially different operation) execute
back-to-back from one unscoped context, with no per-domain access check anywhere
in the path. Nothing here stops the same call pattern from mixing a KYC update
into a balance enquiry, or from touching any customer's account given any prompt
that mentions their ID. Concretely, this one agent now has:

- **Cognitive overload** — the system prompt already has to describe all six
  tools across three unrelated domains, and it only gets worse as more tools are
  added.
- **No separation of concern** — an Accounts query and a KYC update share the
  same agent context; a bug in one domain's tool handling can't be isolated from
  another's.
- **No least privilege** — every request implicitly has access to every tool.
  There is no way to restrict what a given request can do.

Step 4 closes this by splitting the one agent into domain-scoped specialists.
This step's code is not the fix — do not add access restrictions here.

## Regression test

Sidebar has one button per prompt below — re-running all of step 2's prompts:

- `"What is the account balance for customer CUST1001?"` — still works.
- `"What is my cheque book status?"` — previously refused for lacking the
  capability; now the tool exists but is action-only (`request_cheque_book`, not
  a status lookup), so on a fresh/standalone turn the agent should ask for a
  customer ID rather than silently treating a status question as a request to
  dispatch a new cheque book. Worth watching closely — this is a realistic
  tool/intent mismatch. (Mid-conversation, with a customer ID already in context
  from an earlier turn, the agent may instead say it can't check status for that
  customer — also correct, since no status-lookup tool exists either way.)
- `"What is my account balance?"` (no ID) — must still ask for the customer ID.
- `"Can you show me my last 5 transactions?"` — previously refused; now succeeds
  once a customer ID is supplied.

Verified headlessly via `streamlit.testing.v1.AppTest` (clicking each sidebar
button and one manual `chat_input` turn) as well as manually in the browser.

## Run

```
streamlit run 12_2_3_multiple_tools.py
```
