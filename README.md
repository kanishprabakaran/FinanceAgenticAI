# 🧠 Stock Analysis Orchestration System

A multi-agent intelligent orchestration system for performing in-depth stock analysis using the [Phi Framework](https://docs.phi.ai), Groq's LLaMA-4 model, AWS services, and Gemini AI. This system provides analyst recommendations, sentiment evaluations, risk assessments, and strategy recommendations for publicly traded companies with both text and voice input capabilities.

---

## 🚀 Features

- 💹 Analyst insights with recommendation tables  
- 📈 Portfolio strategy guidance (Buy / Hold / Sell)  
- 📰 Latest news aggregation with sources  
- 🧠 Sentiment analysis from headlines  
- ⚠️ Risk analysis from financial fundamentals & news  
- 🎤 Voice input support through AWS Transcribe
- 🤖 Intent classification via Google's Gemini AI
- 🔒 Prompt injection protection and secure .env integration  
- 🤖 Modular, agent-based architecture using `phi.agent`

---

## 🧱 Architecture

### ⚙️ Core Framework
- **Framework:** [`phi`](https://github.com/blackjax-dev/phi)
- **Models:**
  - `Groq` with `meta-llama/llama-4-scout-17b-16e-instruct` (analysis agents)
  - `Google Gemini 1.5 Flash` (intent classification)
- **Environment Loader:** `dotenv` for secure API key handling

### 🔊 Voice Processing
- **Audio Recording:** `sounddevice` for microphone capture
- **AWS Services:**
  - `Amazon S3` for audio file storage
  - `Amazon Transcribe` for speech-to-text conversion

---

## 🧰 Tools

| Tool           | Purpose                                   |
|----------------|-------------------------------------------|
| `YFinanceTools`| Analyst data, fundamentals, news, pricing |
| `DuckDuckGo`   | Public news search, sentiment inputs      |
| `AWS S3`       | Audio file storage                        |
| `AWS Transcribe`| Speech-to-text conversion                |
| `Gemini AI`    | Intent classification                     |

---

## 🤖 Agents

| Agent Name               | Role Description                                                                 |
|--------------------------|----------------------------------------------------------------------------------|
| 📊 `Finance Agent`       | Fetches latest analyst recommendations & fundamentals (tables preferred)         |
| 🧠 `Sentiment Agent`     | Analyzes tone of company news and classifies it (positive/negative/neutral)      |
| ⚠️ `Risk Agent`          | Identifies red flags from financials/news (e.g. debt, lawsuits, scandals)        |
| 📰 `Web Search Agent`    | Retrieves most recent public news from DuckDuckGo                                |
| 📈 `Portfolio Strategy Agent` | Suggests BUY / HOLD / SELL with reasoning and approximate allocations       |
| 🔍 `Intent Classifier`   | Uses Gemini to detect company name and analysis type from user input             |

---

## 🔐 Security Measures

- **Prompt Sanitization:** Blocks injection patterns like `ignore`, `system`, `shutdown`, etc.
- **.env Integration:** API keys securely managed using environment variables
- **Neutralized Input:** Regex filters applied to sanitize all user queries

---

## 📝 Environment Variables

Create a `.env` file with the following variables:

```
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_s3_bucket_name

# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Other API Keys (as needed)
GROQ_API_KEY=your_groq_api_key
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/stock-analysis-orchestrator.git
cd stock-analysis-orchestrator

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run the application
python main.py
```

### Text Input Example:
```
Your query: Should I buy Apple stock?
```

### Voice Input Example:
```
Your query: voice
🎤 Recording for 5 seconds...
🎤 You said: What's the latest news about Tesla?
```

---

## 📋 Requirements

```
phi
groq
google-generativeai
python-dotenv
yfinance
duckduckgo-search
boto3
sounddevice
numpy
wave
```

---

## 🎥 Watch the Stock Analysis Demo

[Watch the Stock Analysis Demo on YouTube](https://youtu.be/BtW9i3CL5KY?si=DOT8iW__zPwX9TWx)
