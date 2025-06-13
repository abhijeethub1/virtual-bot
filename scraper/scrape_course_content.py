#!/usr/bin/env python3
"""
Course Content Scraper for TDS Virtual TA
Scrapes course content from various sources
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseContentScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TDS-Course-Scraper/1.0 (Educational Purpose)'
        })
    
    def scrape_course_page(self, url: str) -> str:
        """Scrape content from a course page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
    
    def scrape_course_materials(self, course_urls: List[str], 
                               output_file: str = "/app/data/tds_content_scraped.txt") -> str:
        """
        Scrape content from multiple course URLs
        
        Args:
            course_urls: List of URLs to scrape
            output_file: Output text file path
        """
        logger.info(f"Starting course content scraping from {len(course_urls)} URLs")
        
        all_content = []
        
        for i, url in enumerate(course_urls):
            logger.info(f"Scraping URL {i+1}/{len(course_urls)}: {url}")
            
            content = self.scrape_course_page(url)
            if content:
                all_content.append(f"=== Content from {url} ===\n\n{content}\n\n")
            
            # Be respectful with rate limiting
            time.sleep(1)
        
        # Combine all content
        combined_content = "\n".join(all_content)
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        logger.info(f"Scraping completed. Saved content to {output_file}")
        return combined_content
    
    def scrape_lecture_transcripts(self, base_url: str, week_range: range = range(1, 9)) -> str:
        """
        Scrape lecture transcripts for specified weeks
        
        Args:
            base_url: Base URL pattern for lectures
            week_range: Range of weeks to scrape
        """
        all_transcripts = []
        
        for week in week_range:
            # This is a placeholder - actual implementation would depend on
            # the specific structure of the course website
            url = f"{base_url}/week-{week}"
            
            logger.info(f"Scraping week {week} content")
            content = self.scrape_course_page(url)
            
            if content:
                all_transcripts.append(f"=== Week {week} Content ===\n\n{content}\n\n")
        
        return "\n".join(all_transcripts)

def create_sample_course_content(output_file: str = "/app/data/tds_content_scraped.txt"):
    """
    Create sample course content for demonstration
    This would be replaced with actual scraping URLs in production
    """
    sample_urls = [
        "https://example.com/tds/week1",
        "https://example.com/tds/week2",
        # Add more URLs as needed
    ]
    
    scraper = CourseContentScraper()
    
    # For demo purposes, we'll just create a sample file
    sample_content = """
Tools in Data Science - Scraped Course Content

This is sample scraped content that would come from actual course pages.
In a real implementation, this scraper would:

1. Connect to the actual course website
2. Navigate through lecture pages
3. Extract text content from each page
4. Clean and format the content
5. Combine into a comprehensive text file

Week 1: Introduction to Python and Data Science
- Python basics and syntax
- Data types and structures
- Introduction to pandas and numpy

Week 2: Data Collection and APIs
- Working with APIs
- Web scraping techniques
- Data formats (JSON, CSV, XML)

[Additional weeks would be scraped and included here...]
"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    logger.info(f"Sample course content created at {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape TDS course content')
    parser.add_argument('--urls', nargs='+', 
                       help='URLs to scrape')
    parser.add_argument('--output', default='/app/data/tds_content_scraped.txt',
                       help='Output file path')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo content instead of scraping')
    
    args = parser.parse_args()
    
    if args.demo or not args.urls:
        create_sample_course_content(args.output)
    else:
        scraper = CourseContentScraper()
        scraper.scrape_course_materials(args.urls, args.output)
