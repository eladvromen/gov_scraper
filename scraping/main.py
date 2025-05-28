import argparse
import os
from scraping.utiac_scraper import UTIACScraper

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='UTIAC Decision Scraper')
    
    parser.add_argument(
        '--start-page', 
        type=int, 
        default=1,
        help='Page number to start scraping from (default: 1)'
    )
    
    parser.add_argument(
        '--end-page', 
        type=int, 
        default=None,
        help='Page number to stop scraping at (default: all pages)'
    )
    
    parser.add_argument(
        '--rate-limit', 
        type=float, 
        default=1.0,
        help='Time in seconds to wait between requests (default: 1.0)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data',
        help='Directory to save scraped data (default: data)'
    )
    
    parser.add_argument(
        '--resume', 
        action='store_true',
        help='Resume scraping from the last page processed'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run in test mode (only scrapes 3 pages)'
    )
    
    return parser.parse_args()

def get_resume_page(output_dir):
    """Get the page to resume from"""
    progress_path = os.path.join(output_dir, 'progress.json')
    if os.path.exists(progress_path):
        import json
        with open(progress_path, 'r') as f:
            progress = json.load(f)
            return progress.get('current_page', 1)
    return 1

def main():
    """Main entry point"""
    args = parse_args()
    
    # Handle resume option
    resume_from_page = None
    if args.resume:
        resume_from_page = get_resume_page(args.output_dir)
        print(f"Resuming from page {resume_from_page}")
    
    # Handle test mode
    if args.test:
        end_page = 3  # Only scrape 3 pages in test mode
    else:
        end_page = args.end_page
    
    # Initialize and run the scraper
    scraper = UTIACScraper(
        rate_limit=args.rate_limit,
        output_dir=args.output_dir,
        resume_from_page=resume_from_page
    )
    
    stats = scraper.scrape_all_decisions(
        start_page=args.start_page if not args.resume else resume_from_page,
        end_page=end_page
    )
    
    # Print summary
    print("\nScraping Summary:")
    print(f"Pages processed: {stats['pages_processed']}")
    print(f"Decisions successfully scraped: {stats['decisions_successful']}")
    print(f"Decisions failed: {stats['decisions_failed']}")
    print(f"Success rate: {stats['decisions_successful'] / (stats['decisions_successful'] + stats['decisions_failed']) * 100:.2f}%")
    print(f"Data saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 