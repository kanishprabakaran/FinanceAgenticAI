import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq 
from phi.tools.yfinance import YFinanceTools 
from phi.tools.duckduckgo import DuckDuckGo
import re

# Load API keys from .env
load_dotenv()

# Initialize Groq model
groq_model = Groq(id="meta-llama/llama-4-scout-17b-16e-instruct")

# Define common tools
ddg_tool = DuckDuckGo()
yfinance_tool = YFinanceTools(
    stock_price=True,
    analyst_recommendations=True,
    stock_fundamentals=True,
    company_news=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[yfinance_tool],
    instructions=["Use tables to display the data", "Focus on analyst recommendations"],
    show_tools_calls=True,
    markdown=True,
)

# Sentiment Agent
sentiment_agent = Agent(
    name="Sentiment Analysis Agent",
    role="Analyze sentiment of recent news and headlines about companies",
    model=groq_model,
    tools=[ddg_tool],
    instructions=[
        "Summarize the sentiment (positive, negative, neutral) from recent news headlines",
        "Justify sentiment with specific headlines",
        "Always include sources"
    ],
    show_tools_calls=True,
    markdown=True,
)

# Risk Agent
risk_agent = Agent(
    name="Risk Analysis Agent",
    role="Identify potential financial or reputational risks",
    model=groq_model,
    tools=[
        ddg_tool,
        YFinanceTools(
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=[
        "Analyze financial fundamentals and news to identify potential risks (e.g., debt, lawsuits, market concerns)",
        "Highlight controversial events or financial red flags",
        "Use tables for risk metrics where possible",
        "Always include sources"
    ],
    show_tools_calls=True,
    markdown=True,
)

# Web Search Agent (optional)
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest news about companies",
    model=groq_model,
    tools=[ddg_tool],
    instructions=["Always include sources", "Display the latest relevant news for the company"],
    show_tools_calls=True,
    markdown=True,
)

# Portfolio Strategy Agent
portfolio_strategy_agent = Agent(
    name="Portfolio Strategy Agent",
    role="Advise on buy/sell/hold decisions for stocks based on multiple signals",
    model=groq_model,
    tools=[yfinance_tool, ddg_tool],
    instructions=[
        "Combine insights from analyst recommendations, sentiment analysis, and risk analysis",
        "If analyst recommendations are strong and sentiment is positive with low risk, suggest BUY",
        "If sentiment is mixed or risks are rising, suggest HOLD or partial SELL",
        "If sentiment is negative and there are major risks or poor fundamentals, suggest SELL",
        "Provide reasons for the decision",
        "Use bullet points or tables for clarity",
        "Suggest approximate percentage allocation if a BUY is recommended"
    ],
    show_tools_calls=True,
    markdown=False,
)

# Controller Agent (Intent Detection + Orchestration)
controller_agent = Agent(
    name="Stock Analysis Orchestrator",
    role="Understand user requests and delegate to appropriate agents for company stock analysis",
    model=groq_model,
    tools=[],  # No tools, just orchestration
    instructions=[
        "If the user asks about stock-related information (buy/sell, price, risk, sentiment, etc.), extract the company name",
        "Then run the following agents in order: Analyst Recommendation, Strategy, News, Sentiment, and Risk",
        "If it's not a stock-related query, just respond accordingly",
    ],
    markdown=True,
    show_tools_calls=False
)

# ----------- EXECUTION SECTION ------------ #
def sanitize_prompt(prompt: str) -> str:
    """Cleans and neutralizes user prompt to prevent prompt injection."""
    blocked_patterns = [
        r"\bignore\b", r"\bsystem\b", r"\binstruction\b",
        r"(?i)(delete|reset|shutdown|format|reboot|inject|execute)",
        r"[#@]{2,}",        # ## system, @@admin
        r"`{2,}",           # Multiline code blocks
        r"\b(prompt|agent)\b",  # Meta terms
        r"\brole:.*",       # Overwrite role
    ]
    for pattern in blocked_patterns:
        prompt = re.sub(pattern, "[REDACTED]", prompt, flags=re.IGNORECASE)
    return prompt.strip()



def full_stock_analysis(company_name: str):
    clean_name = sanitize_prompt(company_name)

    print(f"\n🔍 Analyst Recommendation for {clean_name}:")
    finance_agent.print_response(
        sanitize_prompt(f"What are the latest analyst recommendations for {clean_name}?"),
        stream=True
    )

    print(f"\n📈 Portfolio Strategy Recommendation for {clean_name}:")
    portfolio_strategy_agent.print_response(
        sanitize_prompt(f"Based on current data, should I buy or sell {clean_name} stock?"),
        stream=True
    )

    print(f"\n📰 Latest News about {clean_name}:")
    web_search_agent.print_response(
        sanitize_prompt(f"Show me the latest news about {clean_name}"),
        stream=True
    )

    print(f"\n📊 Sentiment Analysis of {clean_name}:")
    sentiment_agent.print_response(
        sanitize_prompt(f"Analyze the sentiment of recent news about {clean_name}"),
        stream=True
    )

    print(f"\n⚠️ Risk Analysis for {clean_name}:")
    risk_agent.print_response(
        sanitize_prompt(f"Analyze the financial and reputational risk factors for {clean_name}"),
        stream=True
    )

    print(f"\n🔍 Analyst Recommendation for {company_name}:")
    finance_agent.print_response(f"What are the latest analyst recommendations for {company_name}?", stream=True)

    print(f"\n📈 Portfolio Strategy Recommendation for {company_name}:")
    portfolio_strategy_agent.print_response(f"Based on current data, should I buy or sell {company_name} stock?", stream=True)

    print(f"\n📰 Latest News about {company_name}:")
    web_search_agent.print_response(f"Show me the latest news about {company_name}", stream=True)

    print(f"\n📊 Sentiment Analysis of {company_name}:")
    sentiment_agent.print_response(f"Analyze the sentiment of recent news about {company_name}", stream=True)

    print(f"\n⚠️ Risk Analysis for {company_name}:")
    risk_agent.print_response(f"Analyze the financial and reputational risk factors for {company_name}", stream=True)

# Run full analysis for any Stock
full_stock_analysis("Apollo Hospitals Enterprise Limited")