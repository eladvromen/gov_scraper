import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import csv
import random

class UTIACScraper:
    BASE_URL = "https://tribunalsdecisions.service.gov.uk/utiac"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    def __init__(self, 
                 rate_limit: float = 1.0, 
                 output_dir: str = "data",
                 resume_from_page: int = None,
                 max_retries: int = 3):
        """
        Initialize the scraper
        
        Args:
            rate_limit: Time in seconds to wait between requests
            output_dir: Directory to save scraped data
            resume_from_page: Page number to resume scraping from
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.rate_limit = rate_limit
        self.output_dir = output_dir
        self.resume_from_page = resume_from_page
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
        # Create output directories if they don't exist
        os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "text"), exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.setup_logging()
        self.logger.info(f"Initialized scraper with rate limit {rate_limit}s and {max_retries} max retries")
    
    def setup_logging(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger("utiac_scraper")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                f"logs/utiac_scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def get_total_pages(self) -> int:
        """Get the total number of pages from the pagination"""
        try:
            response = self._make_request(self.BASE_URL)
            if not response:
                return 1471  # Default based on provided information
            
            soup = BeautifulSoup(response, 'html.parser')
            
            # Find the pagination links and get the last page number
            pagination = soup.select('.pagination a')
            if pagination:
                # Get the last non-arrow pagination link
                page_numbers = []
                for link in pagination:
                    if link.text.strip().isdigit():
                        page_numbers.append(int(link.text.strip()))
                if page_numbers:
                    return max(page_numbers)
            
            # If we couldn't find pagination, try to calculate from total records
            total_text = soup.find(string=re.compile(r'Displaying Decision .* of \d+ in total'))
            if total_text:
                match = re.search(r'of (\d+) in total', total_text)
                if match:
                    total_records = int(match.group(1))
                    # Assuming 30 records per page
                    return (total_records // 30) + (1 if total_records % 30 > 0 else 0)
            
            # Default fallback
            return 1471
        except Exception as e:
            self.logger.error(f"Error getting total pages: {str(e)}")
            return 1471  # Default based on provided information
    
    def _make_request(self, url: str, retries: int = None) -> Optional[str]:
        """
        Make an HTTP request with retry logic
        
        Args:
            url: The URL to fetch
            retries: Number of retries (defaults to self.max_retries)
            
        Returns:
            The response text or None if all retries failed
        """
        if retries is None:
            retries = self.max_retries
            
        # Add jitter to rate limit to avoid detection
        jitter = random.uniform(-0.2, 0.2) * self.rate_limit
        actual_delay = max(0.1, self.rate_limit + jitter)
        time.sleep(actual_delay)
        
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, timeout=30)
                
                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue
                    
                # Handle other HTTP errors
                response.raise_for_status()
                return response.text
                
            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                if attempt < retries:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.warning(f"Request failed ({str(e)}). Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Failed to fetch {url} after {retries} retries: {str(e)}")
                    return None
    
    def get_page(self, page_num: int) -> Optional[str]:
        """
        Fetch a page of decisions
        
        Args:
            page_num: The page number to fetch
            
        Returns:
            The HTML content or None if an error occurred
        """
        url = f"{self.BASE_URL}?page={page_num}"
        self.logger.info(f"Fetching page {page_num}")
        return self._make_request(url)
    
    def parse_decision_list(self, html: str) -> List[Dict]:
        """
        Parse the list of decisions from a page
        
        Args:
            html: The HTML content of the page
            
        Returns:
            List of decision metadata dictionaries
        """
        soup = BeautifulSoup(html, 'html.parser')
        decisions = []
        
        # Based on the HTML inspection, we can see the decisions are in a table with class 'decisions-table'
        table = soup.find('table', class_='decisions-table')
        if not table:
            self.logger.warning("Could not find decisions table on page")
            return decisions
        
        # Process each row (skip header row)
        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 4:
                # Extract the reference from the link
                ref_col = cols[0]
                ref_link = ref_col.find('a')
                if ref_link:
                    ref_number = ref_link.text.strip()
                    case_url = ref_link.get('href')
                    
                    # Ensure URL is absolute
                    if case_url and not case_url.startswith(('http://', 'https://')):
                        case_url = f"https://tribunalsdecisions.service.gov.uk{case_url}"
                    
                    decision = {
                        'reference_number': ref_number,
                        'status': cols[1].text.strip(),
                        'promulgation_date': cols[2].text.strip(),
                        'country': cols[3].text.strip() if len(cols) > 3 else "",
                        'url': case_url
                    }
                    decisions.append(decision)
        
        return decisions
    
    def get_decision_details(self, url: str) -> Optional[Dict]:
        """
        Fetch and parse a single decision document
        
        Args:
            url: The URL of the decision document
            
        Returns:
            Dictionary of decision details or None if an error occurred
        """
        try:
            self.logger.info(f"Fetching decision {url}")
            response = self._make_request(url)
            if not response:
                return None
                
            soup = BeautifulSoup(response, 'html.parser')
            
            # Initialize the details dictionary
            details = {
                'reference_number': '',
                'case_title': '',
                'appellant_name': '',
                'status': '',
                'hearing_date': '',
                'promulgation_date': '',
                'publication_date': '',
                'last_updated': '',
                'country': '',
                'judges': '',
                'decision_text': '',
                'pdf_url': '',
                'word_url': ''
            }
            
            # Extract the reference number from the h1 tag
            h1 = soup.find('h1')
            if h1:
                details['reference_number'] = h1.text.strip()
            
            # Extract metadata from the definition list
            # Based on the HTML structure we saw in the screenshots
            metadata_items = soup.select('ul.decision-details li')
            for item in metadata_items:
                label = item.find('span', class_='label')
                if label:
                    field_name = label.text.strip().rstrip(':').lower()
                    value = item.text.replace(label.text, '').strip()
                    
                    # Map field names to our dictionary keys
                    if 'case title' in field_name:
                        details['case_title'] = value
                    elif 'appellant name' in field_name:
                        details['appellant_name'] = value
                    elif 'status' in field_name:
                        details['status'] = value
                    elif 'hearing date' in field_name:
                        details['hearing_date'] = value
                    elif 'promulgation date' in field_name:
                        details['promulgation_date'] = value
                    elif 'publication date' in field_name:
                        details['publication_date'] = value
                    elif 'last updated' in field_name:
                        details['last_updated'] = value
                    elif 'country' in field_name:
                        details['country'] = value
                    elif 'judges' in field_name:
                        details['judges'] = value
            
            # Get download URLs
            pdf_link = soup.find('a', string=re.compile('Download a PDF version'))
            if pdf_link:
                details['pdf_url'] = pdf_link['href']
                if not details['pdf_url'].startswith(('http://', 'https://')):
                    details['pdf_url'] = f"https://tribunalsdecisions.service.gov.uk{details['pdf_url']}"
            
            word_link = soup.find('a', string=re.compile('Download a Word document'))
            if word_link:
                details['word_url'] = word_link['href']
                if not details['word_url'].startswith(('http://', 'https://')):
                    details['word_url'] = f"https://tribunalsdecisions.service.gov.uk{details['word_url']}"
            
            # Extract the decision text
            decision_div = soup.find('div', class_='decision-inner')
            if decision_div:
                details['decision_text'] = decision_div.get_text(separator='\n', strip=True)
            
            return details
        except Exception as e:
            self.logger.error(f"Error fetching decision {url}: {str(e)}")
            return None
    
    def save_decision(self, decision: Dict) -> bool:
        """
        Save a decision to JSON and extract text
        
        Args:
            decision: The decision dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            ref_number = decision['reference_number']
            safe_ref = ref_number.replace('/', '_')
            
            # Save to JSON
            json_path = os.path.join(self.output_dir, 'json', f"{safe_ref}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(decision, f, indent=2, ensure_ascii=False)
            
            # Save text separately
            if decision.get('decision_text'):
                text_path = os.path.join(self.output_dir, 'text', f"{safe_ref}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(decision['decision_text'])
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving decision {decision.get('reference_number', 'unknown')}: {str(e)}")
            return False
    
    def update_csv_index(self, decisions: List[Dict]) -> bool:
        """
        Update the CSV index with the latest decisions
        
        Args:
            decisions: List of decision dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            csv_path = os.path.join(self.output_dir, 'csv', 'decisions_index.csv')
            file_exists = os.path.exists(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'reference_number', 'case_title', 'appellant_name', 
                    'status', 'hearing_date', 'promulgation_date', 
                    'publication_date', 'last_updated', 'country', 
                    'judges', 'url', 'pdf_url', 'word_url'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for decision in decisions:
                    # Extract only the fields we want in the CSV
                    row = {field: decision.get(field, '') for field in fieldnames}
                    writer.writerow(row)
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating CSV index: {str(e)}")
            return False
    
    def scrape_page(self, page_num: int) -> Tuple[int, int]:
        """
        Scrape a single page of decisions
        
        Args:
            page_num: The page number to scrape
            
        Returns:
            Tuple of (successful, failed) counts
        """
        html = self.get_page(page_num)
        if not html:
            return 0, 0
        
        decisions_meta = self.parse_decision_list(html)
        self.logger.info(f"Found {len(decisions_meta)} decisions on page {page_num}")
        
        successful = 0
        failed = 0
        detailed_decisions = []
        
        for decision_meta in decisions_meta:
            if decision_meta.get('url'):
                details = self.get_decision_details(decision_meta['url'])
                if details:
                    # Merge metadata with details
                    full_decision = {**decision_meta, **details}
                    if self.save_decision(full_decision):
                        detailed_decisions.append(full_decision)
                        successful += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            else:
                failed += 1
        
        # Update the CSV index
        self.update_csv_index(detailed_decisions)
        
        return successful, failed
    
    def scrape_all_decisions(self, start_page: int = 1, end_page: int = None) -> Dict:
        """
        Main method to scrape all decisions
        
        Args:
            start_page: The page to start scraping from
            end_page: The page to end scraping at (defaults to the last page)
            
        Returns:
            Dictionary with statistics
        """
        # If resuming, use the resume page
        if self.resume_from_page is not None:
            start_page = self.resume_from_page
        
        # Get total pages if end_page not specified
        if end_page is None:
            end_page = self.get_total_pages()
        
        self.logger.info(f"Starting scrape from page {start_page} to {end_page}")
        
        stats = {
            'total_pages': end_page - start_page + 1,
            'pages_processed': 0,
            'decisions_successful': 0,
            'decisions_failed': 0,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        # Create a progress tracking file
        progress_path = os.path.join(self.output_dir, 'progress.json')
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump({
                'start_page': start_page,
                'end_page': end_page,
                'current_page': start_page,
                'start_time': stats['start_time']
            }, f, indent=2)
        
        try:
            for page in range(start_page, end_page + 1):
                self.logger.info(f"Processing page {page} of {end_page}")
                successful, failed = self.scrape_page(page)
                
                stats['pages_processed'] += 1
                stats['decisions_successful'] += successful
                stats['decisions_failed'] += failed
                
                # Update progress
                with open(progress_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'start_page': start_page,
                        'end_page': end_page,
                        'current_page': page + 1,  # Next page to process
                        'start_time': stats['start_time'],
                        'total_successful': stats['decisions_successful'],
                        'total_failed': stats['decisions_failed']
                    }, f, indent=2)
                
                # Log progress every 10 pages
                if page % 10 == 0:
                    self.logger.info(
                        f"Progress: {page}/{end_page} pages, "
                        f"{stats['decisions_successful']} successful, "
                        f"{stats['decisions_failed']} failed"
                    )
        except KeyboardInterrupt:
            self.logger.info("Scraping interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during scraping: {str(e)}")
        finally:
            # Update final stats
            stats['end_time'] = datetime.now().isoformat()
            
            # Save final stats
            stats_path = os.path.join(self.output_dir, 'stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Scraping completed. Stats saved to {stats_path}")
            
            return stats

if __name__ == "__main__":
    # Example usage
    scraper = UTIACScraper(rate_limit=1.0)
    scraper.scrape_all_decisions(start_page=1, end_page=10)  # Just first 10 pages as a test 