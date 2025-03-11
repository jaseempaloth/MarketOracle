from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
import requests
import os

load_dotenv()


def get_stock_symbol(company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}
    
    try:
        res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
        data = res.json()
        return data['quotes'][0]['symbol'] if data['quotes'] else "Unknown"
    except Exception:
        return "Unknown"

web_agent = Agent(
    name="Web Agent",
    model=Groq(id="qwen-2.5-32b"),
    tools=[DuckDuckGo()],
    instructions=[
        "You are a web search agent.",
        "Always include the source of the information in your response.",
    ],
    show_tool_calls=True,
    markdown=True
)

finance_agent = Agent(
    name="Finance Agent",
    model=Groq(id="qwen-2.5-32b"),
    tools=[YFinanceTools(stock_price=True, company_info=True, stock_fundamentals=True,
                        income_statements=True, key_financial_ratios=True, analyst_recommendations=True,
                        company_news=True, technical_indicators=True, historical_prices=True, enable_all=True), get_stock_symbol],
    instructions=[
        "You are a financial analyst.",
        "Use table format to display the data.",
        "If the user asks for the latest stock price, use the get_stock_symbol tool to get the stock symbol and then use the YFinanceTools to get the stock price.",
        "If the user asks for any other information, use the YFinanceTools to get the information.",
    ],
    show_tool_calls=True,
    markdown=True

)

agent_team = Agent(
    name="Agent Team",
    team=[web_agent, finance_agent],
    model=Groq(id="qwen-2.5-32b"),
    instructions=[
        "You are a financial analyst.",
        "Use table format to display the data.",
        "Always include the source of the information in your response.",
        
    ],
    show_tool_calls=True,
    markdown=True
)

# web_agent.print_response('Summarize analyst recommendation and latest news for Tesla', stream=True)
finance_agent.print_response("Compare TESLA and NVIDIA: summarize analyst recommendations and key fundamentals in tables.", stream=True)


# user_input = input("Enter your financial query: ")
# agent_team.print_response(user_input, stream=True)

app = Playground(agents=[finance_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)