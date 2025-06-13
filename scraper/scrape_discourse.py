#!/usr/bin/env python3
"""
Discourse Scraper for TDS Virtual TA
Scrapes discourse posts from a specified date range
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscourseScraper:
    def __init__(self, base_url: str = "https://discourse.onlinedegree.iitm.ac.in", 
                 category: str = "TDS"):
        self.base_url = base_url.rstrip('/')
        self.category = category
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TDS-TA-Bot/1.0 (Educational Purpose)'
        })
    
    def get_category_topics(self, page: int = 0) -> List[Dict[str, Any]]:
        """Get topics from a specific category"""
        try:
            url = f"{self.base_url}/c/{self.category}.json"
            params = {'page': page}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('topic_list', {}).get('topics', [])
        
        except Exception as e:
            logger.error(f"Error fetching category topics: {e}")
            return []
    
    def get_topic_posts(self, topic_id: int) -> Dict[str, Any]:
        """Get all posts from a specific topic"""
        try:
            url = f"{self.base_url}/t/{topic_id}.json"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json()
        
        except Exception as e:
            logger.error(f"Error fetching topic {topic_id}: {e}")
            return {}
    
    def filter_by_date_range(self, topics: List[Dict[str, Any]], 
                           start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Filter topics by date range"""
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        filtered_topics = []
        for topic in topics:
            created_at = topic.get('created_at', '')
            if created_at:
                try:
                    topic_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if start_dt <= topic_dt <= end_dt:
                        filtered_topics.append(topic)
                except ValueError:
                    continue
        
        return filtered_topics
    
    def scrape_discourse_posts(self, start_date: str = "2025-01-01T00:00:00Z", 
                              end_date: str = "2025-04-14T23:59:59Z",
                              output_file: str = "/app/data/discourse.json") -> List[Dict[str, Any]]:
        """
        Scrape discourse posts within date range
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            output_file: Output JSON file path
        """
        logger.info(f"Starting discourse scraping from {start_date} to {end_date}")
        
        all_discourse_data = []
        
        # Get topics from multiple pages
        for page in range(5):  # Limit to 5 pages for demo
            logger.info(f"Scraping page {page + 1}")
            topics = self.get_category_topics(page)
            
            if not topics:
                logger.info("No more topics found")
                break
            
            # Filter by date range
            filtered_topics = self.filter_by_date_range(topics, start_date, end_date)
            
            for topic in filtered_topics:
                topic_id = topic.get('id')
                title = topic.get('title', '')
                
                logger.info(f"Processing topic: {title} (ID: {topic_id})")
                
                # Get all posts for this topic
                topic_data = self.get_topic_posts(topic_id)
                
                if topic_data:
                    posts = []
                    post_stream = topic_data.get('post_stream', {})
                    
                    for post in post_stream.get('posts', []):
                        posts.append({
                            'post_number': post.get('post_number', 0),
                            'content': post.get('cooked', '').strip(),  # HTML content
                            'author': post.get('username', ''),
                            'created_at': post.get('created_at', '')
                        })
                    
                    discourse_entry = {
                        'id': topic_id,
                        'title': title,
                        'url': f"{self.base_url}/t/{topic.get('slug', '')}/{topic_id}",
                        'category': self.category,
                        'created_at': topic.get('created_at', ''),
                        'posts': posts
                    }
                    
                    all_discourse_data.append(discourse_entry)
                
                # Be respectful with rate limiting
                time.sleep(0.5)
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_discourse_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping completed. Saved {len(all_discourse_data)} topics to {output_file}")
        return all_discourse_data

def scrape_with_date_range(start_date: str, end_date: str, 
                          output_file: str = "/app/data/discourse_scraped.json"):
    """
    Convenience function to scrape discourse posts with date range
    
    Usage:
        python scrape_discourse.py --start-date 2025-01-01 --end-date 2025-04-14
    """
    scraper = DiscourseScraper()
    return scraper.scrape_discourse_posts(
        start_date=f"{start_date}T00:00:00Z",
        end_date=f"{end_date}T23:59:59Z",
        output_file=output_file
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape TDS Discourse posts')
    parser.add_argument('--start-date', default='2025-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-04-14', 
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='/app/data/discourse_scraped.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    scrape_with_date_range(args.start_date, args.end_date, args.output)
