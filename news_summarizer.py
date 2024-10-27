import os
import json
from typing import List, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from datetime import datetime
from parser import process_rss_feed
from tqdm import tqdm
import time
import random
import asyncio

class QuotaExceededError(Exception):
    """Custom exception for quota exceeded errors"""
    pass

class NewsSummarizer:
    def __init__(self, api_key: str, output_dir: str = "training_data"):
        self.api_key = api_key
        self.output_dir = output_dir
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Define safety settings
        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_NONE
            }
        ]
        
        # Rate limiting settings
        self.min_delay = 2.0  # Minimum delay between requests in seconds
        self.max_delay = 4.0  # Maximum delay for jitter
        self.max_retries = 3  # Maximum number of retries per article
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    async def wait_with_jitter(self):
        """Wait with random jitter to avoid hitting rate limits"""
        delay = self.min_delay + random.random() * (self.max_delay - self.min_delay)
        await asyncio.sleep(delay)

    async def summarize_with_retry(self, article: Dict[str, Any], retries: int = 0) -> Dict[str, Any]:
        """Attempt to summarize an article with retry logic"""
        try:
            response = await self.model.generate_content_async(
                self.generate_summary_prompt(article),
                safety_settings=self.safety_settings
            )
            
            return {
                "id": hash(article['link'] + article['pubDate']),
                "url": article['link'],
                "title": article['title'],
                "original_content": article['full_content'],
                "summary": response.text,
                "timestamp": article['pubDate'],
                "metadata": {
                    "source": article['link'].split('/')[2],
                    "summarization_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a quota error
            if "429" in error_str or "quota" in error_str.lower() or "exhausted" in error_str.lower():
                if retries < self.max_retries:
                    # Exponential backoff
                    wait_time = (2 ** retries) * 5 + random.uniform(0, 1)
                    print(f"\nQuota exceeded, waiting {wait_time:.1f} seconds before retry {retries + 1}/{self.max_retries}")
                    await asyncio.sleep(wait_time)
                    return await self.summarize_with_retry(article, retries + 1)
                else:
                    raise QuotaExceededError("API quota exceeded after max retries")
            
            print(f"Error summarizing article {article['link']}: {str(e)}")
            return None

    async def process_feed(self, rss_url: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Process an RSS feed and generate summaries for all articles."""
        articles = process_rss_feed(rss_url, hours)
        summaries = []
        
        for article in tqdm(articles, desc="Summarizing articles"):
            try:
                # Wait before making the request
                await self.wait_with_jitter()
                
                summary = await self.summarize_with_retry(article)
                if summary:
                    summaries.append(summary)
                    
                    # Save progress after each successful summary
                    self.save_progress(summaries)
                    
            except QuotaExceededError:
                print(f"\nQuota exceeded for {article['link']}, saving progress and exiting...")
                return summaries
            
        return summaries

    def save_progress(self, summaries: List[Dict[str, Any]]):
        """Save current progress to a temporary file"""
        temp_file = os.path.join(self.output_dir, "summarizer_progress.json")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "creation_date": datetime.now().isoformat(),
                "total_examples": len(summaries),
                "data": summaries
            }, f, indent=2, ensure_ascii=False)

    def save_dataset(self, summaries: List[Dict[str, Any]], filename: str = None):
        """Save the final dataset"""
        if filename is None:
            filename = f"news_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "version": "1.0",
                "creation_date": datetime.now().isoformat(),
                "total_examples": len(summaries),
                "data": summaries
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filepath}")
        return filepath

    def generate_summary_prompt(self, article: Dict[str, Any]) -> str:
        """Generate a prompt for the Gemini API to summarize an article."""
        return f""" You are an AI summarizer. You only summarize articles in bullet points. 
        You do not output any other text. Each bullet point should be a single sentence. 
        Do not use nested bullet points or subpoints. Start each bullet point with a dash (-) 
        followed by a space. Focus on the key points and maintain factual accuracy. 
        You MUST write the summary in the same language as the full article. If the content is in Spanish, write the summary in Spanish, if the content is in English, write the summary in English. The language of the instruction doesn't matter, only the language of the content.
        
        Title: {article['title']}
        
        Content:
        {article['full_content']}
        
        Summary:"""

async def main():
    # Initialize summarizer
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GEMINI_API_KEY environment variable")
    
    summarizer = NewsSummarizer(api_key)
    
    # Load RSS feeds
    with open("sources.json", "r") as f:
        rss_feeds = json.load(f)
    
    all_summaries = []
    
    # Process each feed
    for feed_name, url in rss_feeds.items():
        print(f"\nProcessing feed: {feed_name}")
        try:
            summaries = await summarizer.process_feed(url, hours=24)
            all_summaries.extend(summaries)
        except QuotaExceededError:
            print("Quota exceeded, stopping processing...")
            break
    
    # Save the final dataset
    if all_summaries:
        summarizer.save_dataset(all_summaries)
    else:
        print("No summaries were generated")

if __name__ == "__main__":
    asyncio.run(main())