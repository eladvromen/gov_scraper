# UTIAC Decision Scraper

A Python-based scraper for automatically downloading and organizing UK Immigration and Asylum Tribunal (UTIAC) decisions from the GOV.UK website.

## Features

- Scrapes all ~44,000 legal case records from the UTIAC
- Extracts metadata like reference number, status, dates, country, etc.
- Downloads and organizes the full text content of each decision
- Provides PDF and Word document download links
- Implements proper rate limiting to respect the server
- Includes robust error handling and logging
- Supports resuming scraping if interrupted
- Saves data in multiple formats (JSON, text, CSV)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/utiac-scraper.git
cd utiac-scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To start scraping with default settings:

```bash
python main.py
```

This will:
- Start scraping from page 1
- Continue until all pages are scraped
- Wait 1 second between requests
- Save data to the `data` directory

### Command Line Options

```
usage: main.py [-h] [--start-page START_PAGE] [--end-page END_PAGE] [--rate-limit RATE_LIMIT] [--output-dir OUTPUT_DIR] [--resume] [--test]

UTIAC Decision Scraper

optional arguments:
  -h, --help            show this help message and exit
  --start-page START_PAGE
                        Page number to start scraping from (default: 1)
  --end-page END_PAGE   Page number to stop scraping at (default: all pages)
  --rate-limit RATE_LIMIT
                        Time in seconds to wait between requests (default: 1.0)
  --output-dir OUTPUT_DIR
                        Directory to save scraped data (default: data)
  --resume              Resume scraping from the last page processed
  --test                Run in test mode (only scrapes 3 pages)
```

### Examples

1. Test the scraper with just 3 pages:
```bash
python main.py --test
```

2. Scrape a specific range of pages:
```bash
python main.py --start-page 100 --end-page 150
```

3. Resume a previously interrupted scrape:
```bash
python main.py --resume
```

4. Adjust rate limiting to be more conservative:
```bash
python main.py --rate-limit 2.0
```

## Output Structure

The scraper creates the following directory structure:

```
data/
├── csv/
│   └── decisions_index.csv         # CSV index of all decisions
├── json/
│   ├── PA_01051_2018.json          # Full metadata for each decision
│   ├── HU_07931_2016.json
│   └── ...
├── text/
│   ├── PA_01051_2018.txt           # Plain text of each decision
│   ├── HU_07931_2016.txt
│   └── ...
├── progress.json                   # Scraping progress tracker
└── stats.json                      # Overall scraping statistics
```

## Legal and Ethical Considerations

This scraper is designed to respect the GOV.UK website by:

1. Implementing rate limiting to avoid server overload
2. Following robots.txt guidelines
3. Including proper user agent identification
4. Providing appropriate attribution

All content from GOV.UK is available under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), unless otherwise stated.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 