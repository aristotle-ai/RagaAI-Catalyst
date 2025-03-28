import os
from dotenv import load_dotenv
import openai
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from agents import Agent, Runner, set_tracing_export_api_key

from ragaai_catalyst import RagaAICatalyst, init_tracing
from ragaai_catalyst.tracers import Tracer

load_dotenv()
set_tracing_export_api_key(os.getenv('OPENAI_API_KEY'))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not OPENAI_API_KEY or not YOUTUBE_API_KEY:
    raise EnvironmentError("Please set OPENAI_API_KEY and YOUTUBE_API_KEY in the environment or .env file.")

def initialize_catalyst():
    """Initialize RagaAI Catalyst using environment credentials."""
    catalyst = RagaAICatalyst(
        access_key=os.getenv('CATALYST_ACCESS_KEY'), 
        secret_key=os.getenv('CATALYST_SECRET_KEY'), 
        base_url=os.getenv('CATALYST_BASE_URL')
    )
    
    tracer = Tracer(
        project_name=os.environ.get('PROJECT_NAME', 'email-extraction'),
        dataset_name=os.environ.get('DATASET_NAME', 'email-data'),
        tracer_type="agentic/openai_agents",
    )
    
    init_tracing(catalyst=catalyst, tracer=tracer)

openai.api_key = OPENAI_API_KEY

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def search_video(query: str, channel_url: str = None) -> str:
    """
    Search for a YouTube video by query. If channel_url is provided, restrict the search to that channel.
    Returns the URL of the top matching YouTube video, or an empty string if none found.
    """
    try:
        if channel_url:
            channel_id = None
            if "/channel/" in channel_url:
                channel_id = channel_url.split("/channel/")[1].split("/")[0]
            elif "/user/" in channel_url:
                username = channel_url.split("/user/")[1].split("/")[0]
                channels_response = youtube.channels().list(part="id", forUsername=username).execute()
                if channels_response.get("items"):
                    channel_id = channels_response["items"][0]["id"]
            elif "/@" in channel_url:
                handle = channel_url.split("/@")[1].split("/")[0]
                search_response = youtube.search().list(q=handle, type="channel", part="snippet", maxResults=1).execute()
                if search_response.get("items"):
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
            elif "/c/" in channel_url:
                custom = channel_url.split("/c/")[1].split("/")[0]
                search_response = youtube.search().list(q=custom, type="channel", part="snippet", maxResults=1).execute()
                if search_response.get("items"):
                    channel_id = search_response["items"][0]["snippet"]["channelId"]
            if channel_id:
                search_results = youtube.search().list(q=query, channelId=channel_id, part="snippet", type="video", maxResults=1, order="relevance").execute()
            else:
                search_results = youtube.search().list(q=f"{query}", part="snippet", type="video", maxResults=1, order="relevance").execute()
        else:
            search_results = youtube.search().list(q=query, part="snippet", type="video", maxResults=1, order="relevance").execute()
        items = search_results.get("items", [])
        if not items:
            return "" 
        video_id = items[0]["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        return video_url
    except Exception as e:
        return ""

def get_transcript(video_identifier: str) -> str:
    """
    Retrieve the transcript text for a given YouTube video.
    Accepts a YouTube video URL or video ID.
    Returns the transcript as a single string (empty string if not available).
    """
    try:
        if "youtube.com" in video_identifier or "youtu.be" in video_identifier:
            if "watch?v=" in video_identifier:
                video_id = video_identifier.split("watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_identifier:
                video_id = video_identifier.split("youtu.be/")[1].split("?")[0]
            elif "/shorts/" in video_identifier:
                video_id = video_identifier.split("/shorts/")[1].split("?")[0]
            elif "/embed/" in video_identifier:
                video_id = video_identifier.split("/embed/")[1].split("?")[0]
            else:
                video_id = video_identifier.rstrip("/").split("/")[-1]
        else:
            video_id = video_identifier
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript_text = " ".join([entry.get("text", "") for entry in transcript_list])
        return transcript_text
    except Exception as e:
        return ""

summarizer_agent = Agent(
    name="Summarizer",
    instructions=(
        "You are an assistant that summarizes YouTube video transcripts. "
        "Provide a clear and concise summary of the video's content in a single paragraph. "
        "Make sure the summary is engaging and easy to understand."
    )
)

def main():
    if os.getenv('CATALYST_ACCESS_KEY'):
        initialize_catalyst()
    user_query = input("Enter your query (YouTube URL or search term): ").strip()
    if not user_query:
        print("No query provided. Please enter a YouTube link or search query.")
        return

    channel_url = None
    search_query = None
    video_url = None

    if ("youtube.com/watch" in user_query) or ("youtu.be/" in user_query) or ("youtube.com/shorts/" in user_query) or ("youtube.com/embed/" in user_query):
        video_url = user_query
    elif user_query.startswith("http") and "youtube.com/" in user_query and " - " in user_query:
        parts = user_query.split(" - ", 1)
        channel_url = parts[0].strip()
        search_query = parts[1].strip()
    elif user_query.startswith("http") and "youtube.com/" in user_query and " " in user_query and "-" not in user_query:
        parts = user_query.split(" ", 1)
        channel_url = parts[0].strip()
        search_query = parts[1].strip()
    elif ("youtube.com/channel/" in user_query or "youtube.com/c/" in user_query or 
          "youtube.com/user/" in user_query or "youtube.com/@" in user_query):
        clarifier_agent = Agent(
            name="Clarifier",
            instructions="You are an assistant that asks the user a single clarifying question when their request is ambiguous or incomplete."
        )
        prompt = (
            f"The user only provided a channel URL ({user_query}) without specifying what they want. "
            "Ask a concise question to clarify what they are looking for on this channel."
        )
        clarification_result = Runner.run_sync(clarifier_agent, prompt)
        clarifying_question = clarification_result.final_output.strip()
        followup = input(clarifying_question + " ").strip()
        if not followup:
            print("No details provided. Unable to determine what content to summarize.")
            return
        channel_url = user_query
        search_query = followup
    else:
        search_query = user_query

    if video_url is None:
        if search_query:
            query_terms = search_query
            for term in ["summary of", "Summary of", "summarize", "Summarize"]:
                query_terms = query_terms.replace(term, "")
            query_terms = query_terms.strip()
        else:
            query_terms = ""
        video_url = search_video(query_terms, channel_url)
        if not video_url:
            print("No relevant video could be found for the given query. Please try a different query.")
            return

    transcript_text = get_transcript(video_url)
    if not transcript_text:
        print("Could not retrieve the transcript for the video (it may be unavailable or unsupported).")
        return

    try:
        result = Runner.run_sync(summarizer_agent, transcript_text)
        summary_text = result.final_output.strip()
    except Exception as e:
        print("An error occurred while summarizing the video content.")
        return

    print("\nSummary:\n" + summary_text)
    print("\nVideo Link: " + video_url)

if __name__ == "__main__":
    main()
    ## Sample user inputs:
    ## https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ## Steve Jobs Stanford commencement speech
    ## https://youtube.com/@veritasium - time dilation explanation