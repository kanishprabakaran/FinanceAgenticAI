# FinanceAgenticAI
# 🧠 Stock Analysis Orchestration System

A multi-agent intelligent orchestration system for performing in-depth stock analysis using the [Phi Framework](https://docs.phi.ai) and Groq’s LLaMA-4 model. This system provides analyst recommendations, sentiment evaluations, risk assessments, and strategy recommendations for publicly traded companies.

---

## 🚀 Features

- 💹 Analyst insights with recommendation tables  
- 📈 Portfolio strategy guidance (Buy / Hold / Sell)  
- 📰 Latest news aggregation with sources  
- 🧠 Sentiment analysis from headlines  
- ⚠️ Risk analysis from financial fundamentals & news  
- 🔒 Prompt injection protection and secure .env integration  
- 🤖 Modular, agent-based architecture using `phi.agent`

---

## 🧱 Architecture

### ⚙️ Core Framework

- **Framework:** [`phi`](https://github.com/blackjax-dev/phi)
- **Model:** `Groq` with `meta-llama/llama-4-scout-17b-16e-instruct`
- **Environment Loader:** `dotenv` for secure API key handling

---

## 🧰 Tools

| Tool         | Purpose                                   |
|--------------|-------------------------------------------|
| `YFinanceTools` | Analyst data, fundamentals, news, pricing |
| `DuckDuckGo`    | Public news search, sentiment inputs     |

---

## 🤖 Agents

| Agent Name             | Role Description                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| 📊 `Finance Agent`     | Fetches latest analyst recommendations & fundamentals (tables preferred)         |
| 🧠 `Sentiment Agent`   | Analyzes tone of company news and classifies it (positive/negative/neutral)      |
| ⚠️ `Risk Agent`        | Identifies red flags from financials/news (e.g. debt, lawsuits, scandals)        |
| 📰 `Web Search Agent`  | Retrieves most recent public news from DuckDuckGo                                 |
| 📈 `Portfolio Strategy Agent` | Suggests BUY / HOLD / SELL with reasoning and approximate allocations |
| 🎛️ `Controller Agent` | Delegates task flow across other agents based on user intent                     |

---

## 🔐 Security Measures

- **Prompt Sanitization:** Blocks injection patterns like `ignore`, `system`, `shutdown`, etc.
- **.env Integration:** API keys securely managed using environment variables.
- **Neutralized Input:** Regex filters applied to sanitize all user queries.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/stock-analysis-orchestrator.git
cd stock-analysis-orchestrator
pip install -r requirements.txt
Create a .env file:

env
Copy
Edit
# .env
GROQ_API_KEY=your_groq_api_key_here
▶️ How to Run
python
Copy
Edit
from main import full_stock_analysis

full_stock_analysis("Apollo Hospitals Enterprise Limited")
💡 Output:
Analyst Recommendations

Strategy Suggestion

Latest News

Sentiment Classification

Risk Factors

📍 Workflow
mermaid
Copy
Edit
graph LR
A[User Input] --> B[Controller Agent]
B --> C1[Finance Agent]
B --> C2[Portfolio Strategy Agent]
B --> C3[Web Search Agent]
B --> C4[Sentiment Agent]
B --> C5[Risk Agent]
📘 Example Output
bash
Copy
Edit
🔍 Analyst Recommendation for Apollo Hospitals:
... [Table Output]

📈 Portfolio Strategy Recommendation:
BUY -- Strong analyst support, positive sentiment, low risk
Suggested Allocation: 20%

📰 Latest News:
- "Apollo Hospitals sees Q4 profit jump..." [source]
...

📊 Sentiment:
Overall Sentiment: Positive
Key Headlines: ...
...

⚠️ Risk:
Low financial risk
No legal controversies detected
...
🧩 Future Improvements
Frontend using Streamlit or Next.js

Integrate with real-time trading APIs (e.g. Zerodha, Alpaca)

Add ML-based risk scoring module

📄 License
MIT License © 2025 [Your Name or Organization]

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

