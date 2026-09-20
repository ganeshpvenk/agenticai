import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# CrewAI basics: an Agent has a role/goal/backstory and optional tools,
# a Task describes what that agent should produce, and a Crew runs the
# tasks against the agents. This Agent -> Task -> Crew shape is the
# "Basic Crew" every other example in this directory builds on top of
# - later files add orchestration (Flow), delegation (hierarchical
# Process), human approval, memory/knowledge, structured outputs and
# guardrails around this same core.
#
# Every example in 13_crew_ai/ shares one theme: real stock/company
# data pulled live from Yahoo Finance via yfinance, instead of a
# hand-written or hardcoded scenario.
# =====================================================================


@tool("StockSnapshot")
def get_stock_snapshot(ticker: str) -> str:
    """Fetch a live snapshot of key financial metrics for a stock ticker from Yahoo Finance."""
    info = yf.Ticker(ticker).info
    if not info or info.get("shortName") is None:
        return f"No data found for ticker '{ticker}'."
    return (
        f"{info.get('shortName')} ({ticker.upper()})\n"
        f"Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}\n"
        f"Price: {info.get('currentPrice', info.get('regularMarketPrice', 'n/a'))}\n"
        f"Trailing P/E: {info.get('trailingPE', 'n/a')} | Forward P/E: {info.get('forwardPE', 'n/a')}\n"
        f"Revenue growth (YoY): {info.get('revenueGrowth', 'n/a')}\n"
        f"Profit margin: {info.get('profitMargins', 'n/a')}\n"
        f"52-week range: {info.get('fiftyTwoWeekLow', 'n/a')} - {info.get('fiftyTwoWeekHigh', 'n/a')}\n"
        f"Analyst recommendation: {info.get('recommendationKey', 'n/a')} "
        f"(mean target: {info.get('targetMeanPrice', 'n/a')})"
    )


analyst = Agent(
    role="Financial Data Analyst",
    goal="Fetch and factually summarize a company's current financial snapshot",
    backstory="You pull live market data and report it plainly, without adding opinions.",
    tools=[get_stock_snapshot],
    llm="gpt-4o-mini",
    verbose=True,
)

writer = Agent(
    role="Investment Writer",
    goal="Turn a raw financial snapshot into a short, plain-English brief for a retail investor",
    backstory="You explain financial data clearly, without jargon, in 3-4 sentences.",
    llm="gpt-4o-mini",
    verbose=True,
)

fetch_task = Task(
    description="Fetch the current financial snapshot for {ticker} using the StockSnapshot tool.",
    expected_output="The raw snapshot data for the ticker",
    agent=analyst,
)

brief_task = Task(
    description="Using the snapshot, write a short plain-English investor brief for {ticker}.",
    expected_output="A 3-4 sentence plain-English summary a non-expert could understand",
    agent=writer,
    context=[fetch_task],
)

crew = Crew(
    agents=[analyst, writer],
    tasks=[fetch_task, brief_task],
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff(inputs={"ticker": "AAPL"})
    print("\n" + "=" * 70)
    print("INVESTOR BRIEF")
    print("=" * 70)
    print(result)
