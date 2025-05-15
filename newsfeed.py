import os
import json
import uuid
import time
import io
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2 import service_account
from dotenv import load_dotenv
import isodate
import boto3
from botocore.exceptions import ClientError
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import google.generativeai as genai
import schedule
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        'GEMINI_API_KEY',
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'S3_BUCKET_NAME',
        'YOUTUBE_API_KEY',
        'News_Sheet_ID'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate Google Sheets credentials file
    if not os.path.exists('news_credentials.json'):
        logger.error("Google Sheets credentials file 'news_credentials.json' not found")
        raise FileNotFoundError("Google Sheets credentials file not found")

# Initialize services
try:
    validate_environment()
    
    # Initialize Gemini
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

    # Initialize boto3 session
    boto3_session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=AWS_REGION
    )

    # YouTube Data API Configuration
    youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

    # Google Sheets Configuration
    SHEETS_CREDENTIALS_FILE = 'news_credentials.json'
    SHEET_ID = os.getenv('News_Sheet_ID')
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

def search_youtube_news_videos():
    """Search for recent world news videos in English under 2 minutes from trusted channels."""
    try:
        published_after = (datetime.utcnow() - timedelta(days=1)).isoformat() + "Z"
        
        trusted_channels = [
            "UCupvZG-5ko_eiXAupbDfxWw",  # CNN
            "UCBi2mrWuNuyYy4gbM6fU18Q",  # BBC News
            "UCeY0bbntWzzVIaj2z3QigXg",  # NBC News
            "UC16niRr50-MSBwiO3YDb3RA",  # BBC World News
            "UCNye-wNBqNL5ZzHSJj3l8Bg",  # Al Jazeera English
            "UCIRYBXDze5krPDzAEOxFGVA",  # Reuters
            "UCJg9wBPyKMNA5sRDnvzmkdg",  # Sky News
            "UCUMZ7gohGI9HcU9VNsr2FJQ",  # Bloomberg Global News
            "UChqUTb7kYRX8-EiaN3XFrSQ",  # Reuters Verified
            "UCXIJgqnII2ZOINSWNOGFThA"   # Fox News
        ]
        
        all_videos = []
        
        for channel_id in trusted_channels:
            try:
                channel_videos = youtube.search().list(
                    channelId=channel_id,
                    part="id,snippet",
                    maxResults=5,
                    type="video",
                    videoDuration="short",
                    q="world news",
                    relevanceLanguage="en",
                    publishedAfter=published_after,
                    order="viewCount"
                ).execute()
                
                for item in channel_videos.get('items', []):
                    video_id = item['id']['videoId']
                    
                    video_details = youtube.videos().list(
                        part="contentDetails,statistics",
                        id=video_id
                    ).execute()
                    
                    if not video_details['items']:
                        continue
                        
                    duration = isodate.parse_duration(
                        video_details['items'][0]['contentDetails']['duration']
                    )
                    
                    if duration <= timedelta(minutes=2):
                        view_count = int(video_details['items'][0]['statistics'].get('viewCount', 0))
                        
                        all_videos.append({
                            'id': video_id,
                            'url': f"https://www.youtube.com/watch?v={video_id}",
                            'title': item['snippet']['title'],
                            'published_at': item['snippet']['publishedAt'],
                            'channel_title': item['snippet']['channelTitle'],
                            'view_count': view_count
                        })
                
                logger.info(f"Found {len(channel_videos.get('items', []))} videos from channel {channel_id}")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Error searching channel {channel_id}: {str(e)}")
                continue
        
        all_videos.sort(key=lambda x: x['view_count'], reverse=True)
        return all_videos[:5]
    
    except Exception as e:
        logger.error(f"Error searching YouTube videos: {str(e)}")
        return []

def get_youtube_transcript(video_id):
    """Get transcript directly from YouTube using youtube-transcript-api."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = ' '.join([item['text'] for item in transcript_list])
        logger.info(f"Successfully retrieved transcript for video {video_id}")
        return full_transcript
        
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        logger.warning(f"No transcript available for video {video_id}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving transcript for video {video_id}: {str(e)}")
        return None

def download_audio(video_url):
    """Download audio from YouTube video and return as bytes."""
    try:
        yt = YouTube(video_url)
        audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
        
        if not audio_stream:
            logger.warning("No audio stream found for video")
            return None, None
        
        file_name = f"news_audio_{uuid.uuid4().hex}.mp3"
        buffer = io.BytesIO()
        audio_stream.stream_to_buffer(buffer)
        buffer.seek(0)
        
        # Check file size (limit to 10MB)
        if buffer.getbuffer().nbytes > 10 * 1024 * 1024:
            logger.warning("Audio file too large")
            return None, None
            
        logger.info(f"Downloaded audio from: {yt.title}")
        return buffer.getvalue(), file_name
        
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        return None, None

def upload_to_s3(file_data, bucket_name, file_name):
    """Upload file data to S3 bucket."""
    try:
        s3_client = boto3_session.client('s3')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_data
        )
        logger.info(f"Successfully uploaded {file_name} to S3 bucket {bucket_name}")
        return True
        
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False

def cleanup_s3_files(file_names):
    """Delete files from S3 bucket."""
    try:
        s3_client = boto3_session.client('s3')
        objects = [{'Key': file_name} for file_name in file_names]
        if objects:
            s3_client.delete_objects(
                Bucket=S3_BUCKET_NAME,
                Delete={'Objects': objects}
            )
            logger.info(f"Deleted {len(objects)} files from S3")
    except ClientError as e:
        logger.error(f"Error cleaning up S3 files: {str(e)}")

def transcribe_audio(file_name, s3_uri):
    """Transcribe audio using Amazon Transcribe."""
    try:
        transcribe_client = boto3_session.client('transcribe')
        job_name = f"news_transcription_{uuid.uuid4().hex}"
        
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp3',
            LanguageCode='en-US',
            OutputBucketName=S3_BUCKET_NAME
        )
        
        max_tries = 10
        while max_tries > 0:
            max_tries -= 1
            job = transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            job_status = job['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                break
                
            logger.info("Waiting for transcription to complete...")
            time.sleep(30)
        
        if job_status == 'COMPLETED':
            s3_client = boto3_session.client('s3')
            output_file = f"{job_name}.json"
            
            obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=output_file)
            transcription_data = json.loads(obj['Body'].read().decode('utf-8'))
            transcription = transcription_data['results']['transcripts'][0]['transcript']
            
            logger.info("Transcription completed successfully")
            return transcription, [file_name, output_file]
        else:
            logger.error(f"Transcription job failed: {job_status}")
            return None, [file_name]
            
    except ClientError as e:
        logger.error(f"Error in transcription: {str(e)}")
        return None, [file_name]

def summarize_with_gemini(text):
    """Summarizes text using Gemini model."""
    try:
        prompt = f"""
        You are a news summarization expert. Summarize the following news content 
        in 3-4 concise bullet points, focusing on key facts and main points:
        
        {text}
        
        Provide the summary in this format:
        - [Bullet point 1]
        - [Bullet point 2]
        - [Bullet point 3]
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error summarizing with Gemini: {str(e)}")
        return None

def update_google_sheet(video_data, summary, transcript_method):
    """Updates Google Sheet with video data, summary, and transcript method used."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SHEETS_CREDENTIALS_FILE, scopes=SCOPES)
        
        service = build('sheets', 'v4', credentials=creds)
        
        values = [
            [
                video_data['published_at'],
                video_data['title'],
                video_data['url'],
                summary,
                transcript_method,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        ]
        
        body = {'values': values}
        
        result = service.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range="Sheet1!A:F",
            valueInputOption="USER_ENTERED",
            body=body
        ).execute()
        
        logger.info(f"Updated Google Sheet: {result.get('updates').get('updatedCells')} cells updated")
        return True
    except Exception as e:
        logger.error(f"Error updating Google Sheet: {str(e)}")
        return False

def process_news_videos():
    """Main function to process news videos."""
    start_time = datetime.now()
    logger.info(f"Starting news processing at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    videos = search_youtube_news_videos()
    
    if not videos:
        logger.warning("No suitable videos found")
        return
    
    processed_count = 0
    files_to_cleanup = []
    
    for video in videos:
        logger.info(f"Processing video: {video['title']}")
        video_id = video['id']
        
        transcript = get_youtube_transcript(video_id)
        transcript_method = "YouTube Transcript API"
        
        if not transcript:
            logger.info("No YouTube transcript available. Falling back to AWS transcription...")
            transcript_method = "AWS Transcribe"
            
            audio_data, audio_file_name = download_audio(video['url'])
            if not audio_data:
                logger.warning("Failed to download audio. Skipping video")
                continue
                
            if not upload_to_s3(audio_data, S3_BUCKET_NAME, audio_file_name):
                logger.warning("Failed to upload to S3. Skipping video")
                continue
                
            s3_uri = f"s3://{S3_BUCKET_NAME}/{audio_file_name}"
            transcript, s3_files = transcribe_audio(audio_file_name, s3_uri)
            files_to_cleanup.extend(s3_files)
            
            if not transcript:
                logger.warning("Failed to transcribe audio. Skipping video")
                continue
        
        summary = summarize_with_gemini(transcript)
        if not summary:
            logger.warning("Failed to generate summary. Skipping video")
            continue
            
        logger.info(f"Summary:\n{summary}")
        
        if update_google_sheet(video, summary, transcript_method):
            logger.info("Successfully updated Google Sheet")
            processed_count += 1
        else:
            logger.warning("Failed to update Google Sheet")
        
        time.sleep(2)  # Rate limiting between videos
    
    # Cleanup S3 files
    if files_to_cleanup:
        cleanup_s3_files(files_to_cleanup)
    
    end_time = datetime.now()
    logger.info(f"Completed processing run at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Successfully processed {processed_count} out of {len(videos)} videos")
    logger.info(f"Processing took {(end_time - start_time).total_seconds()} seconds")

def schedule_jobs():
    """Set up scheduled jobs."""
    try:
        schedule.every(1).hours.do(process_news_videos)
        
        logger.info(f"Scheduler initialized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("News processing will run every hour")
        
        # Run once immediately at startup
        process_news_videos()
        
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
        time.sleep(300)  # Wait 5 minutes before retrying
        schedule_jobs()  # Retry scheduling

if __name__ == "__main__":
    try:
        schedule_jobs()
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
