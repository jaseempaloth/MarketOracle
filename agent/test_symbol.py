import requests
import random

# List of common User-Agents
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Safari on M3 Mac
    "Mozilla/5.0 (Macintosh; Apple M3 Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Chrome on M3 Mac
    "Mozilla/5.0 (Macintosh; Apple M3 Mac OS X 14_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    # Firefox on M3 Mac
    "Mozilla/5.0 (Macintosh; Apple M3 Mac OS X 14_3_1; rv:123.0) Gecko/20100101 Firefox/123.0",
    # Edge on M3 Mac
    "Mozilla/5.0 (Macintosh; Apple M3 Mac OS X 14_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Edge/122.0.0.0 Safari/537.36",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
]

def get_stock_symbol(company_name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = random.choice(USER_AGENTS)
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}
    
    try:
        res = requests.get(url=url, params=params, headers={'User-Agent': user_agent})
        data = res.json()
        return data['quotes'][0]['symbol'] if data['quotes'] else "Unknown"
    except Exception as e:
        return "Unknown"

# Test with some well-known companies
test_companies = ["Tesla", "Apple", "Microsoft", "Google", "Netflix"]

print("\nCompany Name to Stock Symbol Conversion:")
print("-" * 40)
for company in test_companies:
    symbol = get_stock_symbol(company)
    print(f"{company:15} -> {symbol}")