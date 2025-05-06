import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq 
from phi.tools.yfinance import YFinanceTools 
from phi.tools.duckduckgo import DuckDuckGo
import re
import google.generativeai as genai
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import time
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime

# Load API keys from .env
load_dotenv()

# Initialize AWS services
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
s3_bucket = os.getenv("S3_BUCKET_NAME")

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

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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

def record_audio(duration=5, sample_rate=44100):
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete")
    return audio_data, sample_rate

def save_audio_to_wav(audio_data, sample_rate, filename="recording.wav"):
    """Save recorded audio to WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return filename

def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket."""
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

def start_transcription_job(bucket_name, audio_file_key, language_code='en-US'):
    """Start an AWS Transcribe job."""
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

def wait_for_transcription(job_name, max_retries=30, wait_interval=5):
    """Wait for transcription job to complete."""
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

def get_transcription_results(bucket_name, job_name):
    """Get transcription results from S3."""
    try:
        result_key = f"{job_name}.json"
        local_filename = f"transcription_{job_name}.json"
        
        s3_client.download_file(bucket_name, result_key, local_filename)
        
        # Parse the JSON file to extract the transcription text
        import json
        with open(local_filename, 'r') as f:
            data = json.load(f)
            transcript = data['results']['transcripts'][0]['transcript']
        
        return transcript
    except Exception as e:
        print(f"Error getting transcription results: {e}")
        return None

def audio_to_text():
    """Record audio, upload to S3, transcribe, and return text."""
    # Record audio
    audio_data, sample_rate = record_audio(duration=5)
    wav_file = save_audio_to_wav(audio_data, sample_rate)
    
    # Upload to S3
    object_name = f"audio_inputs/{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    audio_file_key = upload_to_s3(wav_file, s3_bucket, object_name)
    
    if not audio_file_key:
        return None
    
    # Start transcription
    job_name = start_transcription_job(s3_bucket, audio_file_key)
    if not job_name:
        return None
    
    # Wait for completion
    job_result = wait_for_transcription(job_name)
    if not job_result:
        return None
    
    # Get results
    transcript = get_transcription_results(s3_bucket, job_name)
    return transcript

def classify_intent_and_extract_company(user_input: str) -> dict:
    """Use Gemini to classify the intent and extract company name from natural language."""
    prompt = f"""
    Analyze the following user input and respond in JSON format with:
    1. The company name (extract from text)
    2. The analysis type requested (from these options: 
       - "full_analysis" (general stock/company info)
       - "financials" (financial metrics)
       - "sentiment" (news sentiment)
       - "risk" (risk factors)
       - "recommendation" (buy/sell/hold)
       - "news" (latest news)
    
    User Input: "{user_input}"
    
    Respond ONLY with valid JSON like this:
    {{
        "company": "extracted company name",
        "analysis_type": "detected analysis type"
    }}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        # Extract JSON from response
        json_str = response.text.strip().replace('```json', '').replace('```', '').strip()
        return eval(json_str)
    except Exception as e:
        print(f"Error processing with Gemini: {e}")
        return {"company": "", "analysis_type": "full_analysis"}

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

def run_analysis(company: str, analysis_type: str):
    """Run the appropriate analysis based on the detected type."""
    clean_company = sanitize_prompt(company)
    
    if not clean_company:
        print("⚠️ Could not detect a company name in your query.")
        return
    
    print(f"\n🔍 Analyzing {clean_company} - {analysis_type.upper()}...")
    
    if analysis_type == "full_analysis":
        full_stock_analysis(clean_company)
    elif analysis_type == "financials":
        finance_agent.print_response(
            f"Provide comprehensive financial analysis for {clean_company} including fundamentals, ratios and analyst recommendations",
            stream=True
        )
    elif analysis_type == "sentiment":
        sentiment_agent.print_response(
            f"Analyze the sentiment of recent news about {clean_company}",
            stream=True
        )
    elif analysis_type == "risk":
        risk_agent.print_response(
            f"Analyze the financial and reputational risk factors for {clean_company}",
            stream=True
        )
    elif analysis_type == "recommendation":
        portfolio_strategy_agent.print_response(
            f"Based on current data, should I buy, sell or hold {clean_company} stock? Provide detailed reasoning.",
            stream=True
        )
    elif analysis_type == "news":
        web_search_agent.print_response(
            f"Show me the latest news about {clean_company} from reliable sources",
            stream=True
        )
    else:
        print(f"⚠️ Unknown analysis type: {analysis_type}. Running full analysis instead.")
        full_stock_analysis(clean_company)

def full_stock_analysis(company_name: str):
    """Run complete analysis suite for a company."""
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
    web_search_agent.print_response(
        f"Show me the latest news about {company_name}",
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

def process_user_query(user_input: str):
    """Main function to handle natural language input."""
    # Classify intent and extract company using Gemini
    analysis_request = classify_intent_and_extract_company(user_input)
    
    # Run the appropriate analysis
    run_analysis(
        company=analysis_request.get("company", ""),
        analysis_type=analysis_request.get("analysis_type", "full_analysis")
    )

def main():
    """Main interaction loop."""
    print("💹 Stock Analysis Assistant - Enter your query (e.g., 'Should I buy Apple stock?') or say 'voice' for audio input")
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
