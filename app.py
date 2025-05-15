import os
import subprocess
import json
import re
import time
import wave
from datetime import datetime
from typing import Dict, List

import boto3
import gspread
import numpy as np
import pandas as pd
import sounddevice as sd
import wbgapi as wb
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from newsapi import NewsApiClient
from oauth2client.service_account import ServiceAccountCredentials
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools import Toolkit
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import google.generativeai as genai

# Load API keys from .env
load_dotenv()

# Initialize AWS services
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
s3_bucket = os.getenv("S3_BUCKET_NAME")

try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

    transcribe_client = boto3.client(
        'transcribe',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
except Exception as e:
    print(f"Error initializing AWS clients: {e}")
    s3_client = None
    transcribe_client = None

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    print("Warning: GEMINI_API_KEY not found. Intent classification may fail.")
    gemini_model = None

# Initialize Groq model
groq_model = Groq(id="meta-llama/llama-4-scout-17b-16e-instruct")

# Initialize NewsAPI
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
if NEWSAPI_API_KEY:
    newsapi = NewsApiClient(api_key=NEWSAPI_API_KEY)
else:
    print("Warning: NEWSAPI_API_KEY not found. News fetching may fail.")
    newsapi = None

# Initialize Google Sheets
NEWS_SHEET_ID = os.getenv("NEWS_SHEET_ID")

# Define common tools
ddg_tool = DuckDuckGo()
yfinance_tool = YFinanceTools(
    stock_price=True,
    analyst_recommendations=True,
    stock_fundamentals=True,
    company_news=True
)

# YouTube News Tools (Google Sheets)
class YouTubeNewsTools(Toolkit):
    def __init__(self):
        super().__init__(name="youtube_news_tools")
        try:
            # Load credentials from news_credentials.json
            with open('news_credentials.json') as f:
                creds_dict = json.load(f)
            
            self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            self.creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, self.scope)
            self.client = gspread.authorize(self.creds)
        except Exception as e:
            print(f"Error initializing YouTube News Tools: {e}")
            self.client = None

    def get_top_news(self, limit: int = 5) -> str:
        """Get top news from YouTube curated sheet."""
        if not self.client or not NEWS_SHEET_ID:
            return "YouTube news integration not configured (missing client or NEWS_SHEET_ID)"
        
        try:
            sheet = self.client.open_by_key(NEWS_SHEET_ID).sheet1
            records = sheet.get_all_records()
            
            if not records:
                return "No news found in the YouTube news sheet."
            
            # Sort by views if available
            if 'views' in records[0]:
                records = sorted(records, key=lambda x: int(str(x['views']).replace(',', '')), reverse=True)
            
            result = "📺 Top YouTube News:\n"
            for idx, item in enumerate(records[:limit], 1):
                result += (
                    f"{idx}. {item.get('title', 'No title')}\n"
                    f"   - Channel: {item.get('channel', 'Unknown')}\n"
                    f"   - Views: {item.get('views', 'N/A')}\n"
                    f"   - Published: {item.get('published', 'N/A')}\n"
                    f"   - URL: {item.get('url', '')}\n\n"
                )
            return result
        except Exception as e:
            return f"Error fetching YouTube news: {str(e)}"

# News API Tools
class NewsAPITools(Toolkit):
    def __init__(self):
        super().__init__(name="newsapi_tools")

    def get_top_headlines(self, query: str = None, country: str = 'us', category: str = None) -> str:
        """Get top headlines from NewsAPI."""
        if not newsapi:
            return "NewsAPI not initialized."
        
        try:
            top_headlines = newsapi.get_top_headlines(
                q=query,
                category=category,
                language='en',
                country=country
            )
            if not top_headlines['articles']:
                return "No headlines found for the given criteria."
            
            result = "📰 Top Headlines:\n"
            for article in top_headlines['articles']:
                result += f"- {article['title']} ({article['source']['name']}, {article['publishedAt']})\n"
            return result
        except Exception as e:
            return f"Error fetching headlines: {str(e)}"

    def get_sources(self, category: str = None, language: str = 'en', country: str = 'us') -> str:
        """Get news sources from NewsAPI."""
        if not newsapi:
            return "NewsAPI not initialized."
        
        try:
            sources = newsapi.get_sources(
                category=category,
                language=language,
                country=country
            )
            if not sources['sources']:
                return "No sources found for the given criteria."
            
            result = "News Sources:\n"
            for source in sources['sources']:
                result += f"- {source['name']} ({source['category']}, {source['country']})\n"
            return result
        except Exception as e:
            return f"Error fetching sources: {str(e)}"

    def get_everything(self, query: str, language: str = 'en', sort_by: str = 'publishedAt') -> str:
        """Get all news articles about a topic from NewsAPI."""
        if not newsapi:
            return "NewsAPI not initialized."
        
        try:
            everything = newsapi.get_everything(
                q=query,
                language=language,
                sort_by=sort_by
            )
            if not everything['articles']:
                return f"No articles found for '{query}'."
            
            result = f"Articles about '{query}':\n"
            for article in everything['articles']:
                result += f"- {article['title']} ({article['source']['name']}, {article['publishedAt']})\n"
            return result
        except Exception as e:
            return f"Error fetching news: {str(e)}"

# Macroeconomic Data Tools
class MacroEconomicTools(Toolkit):
    def __init__(self):
        super().__init__(name="macroeconomic_tools")
        self.indicators = {
            'GDP (current US$)': 'NY.GDP.MKTP.CD',
            'GDP growth (annual %)': 'NY.GDP.MKTP.KD.ZG',
            'Inflation, consumer prices (annual %)': 'FP.CPI.TOTL.ZG',
            'Unemployment, total (% of total labor force)': 'SL.UEM.TOTL.ZS',
            'Labor force, total': 'SL.TLF.TOTL.IN',
            'Foreign direct investment, net inflows (% of GDP)': 'BX.KLT.DINV.WD.GD.ZS',
            'Government debt (% of GDP)': 'GC.DOD.TOTL.GD.ZS',
            'Exports of goods and services (% of GDP)': 'NE.EXP.GNFS.ZS',
            'Imports of goods and services (% of GDP)': 'NE.IMP.GNFS.ZS',
            'GDP per capita (current US$)': 'NY.GDP.PCAP.CD'
        }

    def _get_country_code(self, country_name: str) -> str:
        """Get ISO3 country code from name using World Bank API."""
        try:
            # Use wbgapi to search for country
            for economy in wb.economy.list():
                if country_name.lower() in economy['name'].lower():
                    return economy['id']
            return country_name[:3].upper()  # Fallback
        except Exception as e:
            print(f"Error getting country code: {e}")
            return country_name[:3].upper()

    def _fetch_wb_data(self, country_code: str, years: int = 5) -> dict:
        """Fetch raw data from World Bank API."""
        try:
            current_year = datetime.now().year
            time_range = range(current_year - years + 1, current_year + 1)
            
            df = wb.data.DataFrame(
                list(self.indicators.values()),
                country_code,
                time=time_range,
                skipBlanks=True
            )
            
            data = {}
            for indicator_code in df.index.get_level_values('series'):
                indicator_name = next(k for k, v in self.indicators.items() if v == indicator_code)
                values = df.loc[indicator_code].to_dict()
                data[indicator_name] = {
                    'values': {str(year): values.get(str(year)) for year in time_range if pd.notna(values.get(str(year)))},
                    'unit': self._get_unit_from_indicator(indicator_name)
                }
            
            return data
        except Exception as e:
            print(f"Error fetching World Bank data: {e}")
            return {}

    def _get_unit_from_indicator(self, indicator_name: str) -> str:
        """Determine the unit for display based on indicator name."""
        if 'GDP' in indicator_name and 'US$' in indicator_name:
            return 'US$'
        elif '%' in indicator_name:
            return '%'
        elif 'total' in indicator_name.lower():
            return 'people'
        return ''

    def get_macroeconomic_data(self, country: str = 'United States', years: int = 5) -> str:
        """Get comprehensive macroeconomic data for a country."""
        try:
            country_code = self._get_country_code(country)
            if not country_code:
                return f"Could not identify country code for {country}"
            
            data = self._fetch_wb_data(country_code, years)
            if not data:
                return f"No macroeconomic data available for {country} ({country_code})"
            
            output = {
                "country": country,
                "country_code": country_code,
                "time_period": f"Last {years} years",
                "data": data
            }
            
            return json.dumps(output)
        except Exception as e:
            return f"Error getting macroeconomic data: {str(e)}"

# Initialize tools
news_tools = NewsAPITools()
macro_tools = MacroEconomicTools()
youtube_news_tools = YouTubeNewsTools()

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
    model =groq_model,
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

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest news about companies",
    model=groq_model,
    tools=[ddg_tool],
    instructions=["Always include sources", "Display the latest relevant news for the company"],
    show_tools_calls=True,
    markdown=True,
)

# News Agent
news_agent = Agent(
    name="News Agent",
    role="Fetch and analyze news from reliable sources using NewsAPI",
    model=groq_model,
    tools=[news_tools],
    instructions=[
        "Fetch news from reliable sources using NewsAPI",
        "For company-specific news, use get_everything",
        "For general news, use get_top_headlines",
        "Always include publication date and source",
        "Summarize key points from articles",
        "Format results with clear headings and bullet points"
    ],
    show_tools_calls=True,
    markdown=True,
)

# YouTube Top News Agent
youtube_news_agent = Agent(
    name="YouTube Top News Agent",
    role="Fetch and analyze top news from YouTube's trending videos",
    model=groq_model,
    tools=[youtube_news_tools],
    instructions=[
        "Fetch trending news from YouTube's curated spreadsheet",
        "Focus on the most viewed and recent news items",
        "Include channel name, view count, and publication date",
        "Format results with clear headings and bullet points",
        "Highlight any particularly viral or important news items"
    ],
    show_tools_calls=True,
    markdown=True,
)

# Combined News Agent
combined_news_agent = Agent(
    name="Combined News Agent",
    role="Combine and analyze news from both traditional sources and YouTube",
    model=groq_model,
    tools=[news_tools, youtube_news_tools],
    instructions=[
        "When asked for general news or headlines, fetch from both NewsAPI and YouTube",
        "Combine the results into a comprehensive report",
        "Compare coverage between traditional media and YouTube",
        "Note any major stories that appear in both or are unique to each platform",
        "Format results with clear sections for each source type",
        "Include metrics like view counts for YouTube and source credibility for news articles"
    ],
    show_tools_calls=True,
    markdown=True,
)

# Macroeconomic Data Agent
macroeconomic_agent = Agent(
    name="Macroeconomic Data Agent",
    role="Fetch and analyze macroeconomic indicators from World Bank data",
    model=groq_model,
    tools=[macro_tools],
    instructions=[
        "When receiving raw macroeconomic data, transform it into an insightful report with:",
        "1. A brief introduction about the country's economic context",
        "2. Clean tables showing each indicator's values over time",
        "3. Analysis of trends (increasing/decreasing/stagnant) for each indicator",
        "4. Highlight any concerning or positive trends",
        "5. Compare key ratios (like debt-to-GDP) to healthy benchmarks",
        "6. Format numbers properly (thousands separators, decimal places)",
        "7. For missing data points, note they're unavailable",
        "8. Conclude with overall economic health assessment",
        "",
        "Example format:",
        "## Economic Report: [Country] ([Country Code])",
        "### GDP Trends",
        "| Year | GDP (current US$) | GDP Growth (%) |",
        "|------|-------------------|----------------|",
        "| 2023 | $1.23 trillion    | 2.5%           |",
        "...",
        "### Analysis:",
        "- GDP has grown steadily at average of 2.3% annually...",
        "- Debt-to-GDP ratio at 65% exceeds recommended 60% threshold...",
        "",
        "### Overall Assessment: [Positive/Neutral/Cautious/Negative]"
    ],
    show_tools_calls=True,
    markdown=True
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

def record_audio(duration: int = 5, sample_rate: int = 44100) -> tuple:
    """Record audio from microphone."""
    try:
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        print("Recording complete")
        return audio_data, sample_rate
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None, None

def save_audio_to_wav(audio_data: np.ndarray, sample_rate: int, filename: str = "recording.wav") -> str:
    """Save recorded audio to WAV file."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return filename
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None

def upload_to_s3(file_path: str, bucket_name: str, object_name: str = None) -> str:
    """Upload a file to an S3 bucket."""
    if not s3_client:
        print("S3 client not initialized.")
        return None
    
    if object_name is None:
        object_name = os.path.basename(file_path)
    
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File {file_path} uploaded to {bucket_name}/{object_name}")
        return object_name
    except NoCredentialsError:
        print("AWS credentials not available")
        return None
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return None

def start_transcription_job(bucket_name: str, audio_file_key: str, language_code: str = 'en-US') -> str:
    """Start an AWS Transcribe job."""
    if not transcribe_client:
        print("Transcribe client not initialized.")
        return None
    
    job_name = f"transcribe_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': f"s3://{bucket_name}/{audio_file_key}"},
            MediaFormat='wav',
            LanguageCode=language_code,
            OutputBucketName=bucket_name
        )
        print(f"Transcription job started: {job_name}")
        return job_name
    except ClientError as e:
        print(f"Error starting transcription job: {e}")
        return None

def wait_for_transcription(job_name: str, max_retries: int = 30, wait_interval: int = 5) -> dict:
    """Wait for transcription job to complete."""
    if not transcribe_client:
        print("Transcribe client not initialized.")
        return None
    
    print("Waiting for transcription to complete...")
    for _ in range(max_retries):
        try:
            response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            status = response['TranscriptionJob']['TranscriptionJobStatus']
            
            if status == 'COMPLETED':
                print("Transcription completed successfully")
                return response
            elif status == 'FAILED':
                print("Transcription job failed")
                return None
            
            time.sleep(wait_interval)
        except ClientError as e:
            print(f"Error checking transcription status: {e}")
            return None
    
    print("Transcription timed out")
    return None

def get_transcription_results(bucket_name: str, job_name: str) -> str:
    """Get transcription results from S3."""
    if not s3_client:
        print("S3 client not initialized.")
        return None
    
    try:
        result_key = f"{job_name}.json"
        local_filename = f"transcription_{job_name}.json"
        
        s3_client.download_file(bucket_name, result_key, local_filename)
        
        with open(local_filename, 'r') as f:
            data = json.load(f)
            transcript = data['results']['transcripts'][0]['transcript']
        
        return transcript
    except Exception as e:
        print(f"Error getting transcription results: {e}")
        return None

def audio_to_text() -> str:
    """Record audio, upload to S3, transcribe, and return text."""
    audio_data, sample_rate = record_audio(duration=5)
    if audio_data is None:
        return None
    
    wav_file = save_audio_to_wav(audio_data, sample_rate)
    if not wav_file:
        return None
    
    object_name = f"audio_inputs/{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    audio_file_key = upload_to_s3(wav_file, s3_bucket, object_name)
    if not audio_file_key:
        return None
    
    job_name = start_transcription_job(s3_bucket, audio_file_key)
    if not job_name:
        return None
    
    job_result = wait_for_transcription(job_name)
    if not job_result:
        return None
    
    transcript = get_transcription_results(s3_bucket, job_name)
    return transcript

def classify_intent_and_extract_entities(user_input: str) -> Dict[str, List[Dict[str, str]]]:
    """Use Gemini to classify the intent and extract multiple entities and analysis types."""
    if not gemini_model:
        print("Gemini model not initialized. Using fallback.")
        return {"requests": [{"entity": user_input, "analysis_type": "full_analysis"}]}
    
    prompt = f"""
    Analyze this query and respond with JSON containing:
    1. "requests": An array of objects, each containing:
       - "entity": A company or country name
       - "analysis_type": The type of analysis requested
    
    Analysis types include:
    - "financials" (company financial metrics)
    - "sentiment" (news sentiment)
    - "risk" (risk factors)
    - "recommendation" (buy/sell/hold)
    - "news" (latest news)
    - "macroeconomic" (GDP, inflation etc.)
    - "full_analysis" (default for companies)
    - "top_headlines" (general news headlines)
    
    For example, if the query is "Show me Apple financials and news about Tesla":
    {{
        "requests": [
            {{
                "entity": "Apple",
                "analysis_type": "financials"
            }},
            {{
                "entity": "Tesla",
                "analysis_type": "news"
            }}
        ]
    }}
    
    For "GDP and news of Pakistan":
    {{
        "requests": [
            {{
                "entity": "Pakistan",
                "analysis_type": "macroeconomic"
            }},
            {{
                "entity": "Pakistan",
                "analysis_type": "news"
            }}
        ]
    }}
    
    For "show me top headlines":
    {{
        "requests": [
            {{
                "entity": "general",
                "analysis_type": "top_headlines"
            }}
        ]
    }}
    
    If only an entity is given with no specific analysis, use "full_analysis" for companies and "macroeconomic" for countries.
    
    User Input: "{user_input}"
    
    Respond ONLY with valid JSON.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return {"requests": [{"entity": user_input, "analysis_type": "full_analysis"}]}

def sanitize_prompt(prompt: str) -> str:
    """Cleans and neutralizes user prompt to prevent prompt injection."""
    blocked_patterns = [
        r"\bignore\b", r"\bsystem\b", r"\binstruction\b",
        r"(?i)(delete|reset|shutdown|format|reboot|inject|execute)",
        r"[#@]{2,}",
        r"`{2,}",
        r"\b(prompt|agent)\b",
        r"\brole:.*",
    ]
    for pattern in blocked_patterns:
        prompt = re.sub(pattern, "[REDACTED]", prompt, flags=re.IGNORECASE)
    return prompt.strip()

def run_analysis(entity: str, analysis_type: str) -> None:
    """Run the appropriate analysis based on the detected type."""
    clean_entity = sanitize_prompt(entity)
    
    if not clean_entity:
        print("⚠️ Could not detect a company or country name in your query.")
        return
    
    print(f"\n🔍 Analyzing {clean_entity} - {analysis_type.upper()}...")
    
    try:
        if analysis_type == "full_analysis":
            full_stock_analysis(clean_entity)
        elif analysis_type == "financials":
            finance_agent.print_response(
                f"Provide comprehensive financial analysis for {clean_entity} including fundamentals, ratios and analyst recommendations",
                stream=True
            )
        elif analysis_type == "sentiment":
            sentiment_agent.print_response(
                f"Analyze the sentiment of recent news about {clean_entity}",
                stream=True
            )
        elif analysis_type == "risk":
            risk_agent.print_response(
                f"Analyze the financial and reputational risk factors for {clean_entity}",
                stream=True
            )
        elif analysis_type == "recommendation":
            portfolio_strategy_agent.print_response(
                f"Based on current data, should I buy, sell or hold {clean_entity} stock? Provide detailed reasoning.",
                stream=True
            )
        elif analysis_type == "news":
            news_agent.print_response(
                f"Show me the latest news about {clean_entity} from reliable sources",
                stream=True
            )
        elif analysis_type == "top_headlines":
            combined_news_agent.print_response(
                "Provide a comprehensive news report combining top headlines from traditional news sources and YouTube trending videos",
                stream=True
            )
        elif analysis_type == "macroeconomic":
            macroeconomic_agent.print_response(
                f"Provide macroeconomic data for {clean_entity} including GDP, inflation, unemployment, and labor force statistics",
                stream=True
            )
        else:
            print(f"⚠️ Unknown analysis type: {analysis_type}. Running full analysis instead.")
            full_stock_analysis(clean_entity)
    except Exception as e:
        print(f"Error during analysis: {e}")

def full_stock_analysis(company_name: str) -> None:
    """Run complete analysis suite for a company."""
    try:
        print(f"\n🔍 Analyst Recommendation for {company_name}:")
        finance_agent.print_response(
            f"What are the latest analyst recommendations for {company_name}?",
            stream=True
        )

        print(f"\n📈 Portfolio Strategy Recommendation for {company_name}:")
        portfolio_strategy_agent.print_response(
            f"Based on current data, should I buy or sell {company_name} stock?",
            stream=True
        )

        print(f"\n📰 Latest News about {company_name}:")
        news_agent.print_response(
            f"Show me the latest news about {company_name} from reliable sources",
            stream=True
        )

        print(f"\n📊 Sentiment Analysis of {company_name}:")
        sentiment_agent.print_response(
            f"Analyze the sentiment of recent news about {company_name}",
            stream=True
        )

        print(f"\n⚠️ Risk Analysis for {company_name}:")
        risk_agent.print_response(
            f"Analyze the financial and reputational risk factors for {company_name}",
            stream=True
        )
    except Exception as e:
        print(f"Error during full stock analysis: {e}")

def process_user_query(user_input: str) -> None:
    """Main function to handle natural language input with support for multiple requests."""
    analysis_requests = classify_intent_and_extract_entities(user_input)
    
    if not analysis_requests or "requests" not in analysis_requests or not analysis_requests["requests"]:
        print("⚠️ Could not understand your query. Please try again with a clearer request.")
        return
    
    for idx, request in enumerate(analysis_requests["requests"]):
        if idx > 0:
            print("\n" + "="*50 + "\n")
            
        entity = request.get("entity", "")
        analysis_type = request.get("analysis_type", "full_analysis")
        
        if not entity:
            print("⚠️ Could not detect a company or country name in part of your query.")
            continue
            
        run_analysis(entity=entity, analysis_type=analysis_type)

def main() -> None:
    """Main interaction loop."""
    print("💹 Stock Analysis Assistant - Enter your query (e.g., 'Should I buy Apple stock?' or 'GDP of USA') or say 'voice' for audio input")
    print("💡 Pro tip: You can combine multiple requests in one query, like 'Show me Apple financials and news about Tesla' or 'GDP and news of Pakistan'")
    print("📰 For top news, try: 'Show me today's top headlines' or 'What's trending on YouTube?'")
    
    while True:
        try:
            user_input = input("\nYour query (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input:
                continue
                
            if user_input.lower() == 'voice':
                print("\n🎤 Voice input selected. Recording for 5 seconds...")
                transcript = audio_to_text()
                if transcript:
                    print(f"\n🎤 You said: {transcript}")
                    process_user_query(transcript)
                else:
                    print("Failed to process audio input. Please try again or type your query.")
            else:
                process_user_query(user_input)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}. Please try again.")

if __name__ == "__main__":
    main()
