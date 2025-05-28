import json
import pandas as pd
import re
import os
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalCasePreprocessor:
    """
    A comprehensive preprocessor for UK Immigration and Asylum Tribunal cases.
    Handles JSON files containing legal decisions and metadata.
    """
    
    def __init__(self, json_dir: str, output_dir: str = "preprocessing/processed_data"):
        """
        Initialize the preprocessor.
        
        Args:
            json_dir: Directory containing JSON files
            output_dir: Directory to save processed outputs
        """
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define expected metadata fields
        self.metadata_fields = [
            'reference_number', 'status', 'promulgation_date', 'country', 
            'url', 'case_title', 'appellant_name', 'hearing_date', 
            'publication_date', 'last_updated', 'judges', 'pdf_url', 'word_url'
        ]
        
    def clean_decision_text(self, text: str) -> str:
        """
        Clean and preprocess the decision text for better readability and analysis.
        
        Args:
            text: Raw decision text
            
        Returns:
            Cleaned decision text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # 1. Fix common encoding issues
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--') # Em dash
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        
        # 2. Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Leading whitespace on lines
        
        # 3. Clean up common legal document artifacts
        # Remove page headers/footers (common patterns)
        text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)
        
        # 4. Remove tribunal headers and administrative text at the beginning
        header_patterns = [
            r'^.*?IMMIGRATION APPEAL TRIBUNAL\s*\n',
            r'^.*?Date of\s+Hearing:\s*\n.*?\n',
            r'^.*?Date.*?notified:\s*\n.*?\n',
            r'^.*?Before:\s*\n',
            r'^.*?APPELLANT\s*\n',
            r'^.*?RESPONDENT\s*\n',
            r'^.*?DETERMINATION AND REASONS\s*\n',
            r'^.*?Represented by:.*?\n',
            r'^.*?Heard at:.*?\n',
            r'^.*?APPEAL NO:.*?\n',
            r'^.*?On:.*?\n',
        ]
        
        for pattern in header_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 5. Remove case reference lines and administrative info at the start
        text = re.sub(r'^\s*\[?\d{4}\]?\s*UKIAT\s*\d+.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d{1,2}/\d{1,2}/\d{4}\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d{1,2}\s+\w+\s+\d{4}\s*\n', '', text, flags=re.MULTILINE)
        
        # 6. Remove judge names at the beginning (but keep in main text if part of content)
        text = re.sub(r'^.*?(?:Mr Justice|Mrs Justice|Mr|Mrs|Ms)\s+[A-Z][a-z]+.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^.*?(?:The Honourable).*?\n', '', text, flags=re.MULTILINE)
        
        # 7. Remove appellant/respondent name patterns at start
        text = re.sub(r'^.*?(?:and|v\.?)\s*\n.*?(?:SECRETARY OF STATE|HOME DEPARTMENT).*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 8. Clean up case citations and references
        text = re.sub(r'\[(\d{4})\]\s*UKIAT\s*(\d+)', r'[\1] UKIAT \2', text)
        
        # 9. Standardize paragraph numbering
        text = re.sub(r'\n(\d+)\.\s+', r'\n\1. ', text)  # Ensure space after numbered paragraphs
        
        # 10. Remove trailing signatures and administrative text
        # Remove judge signatures at the end
        text = re.sub(r'\n\s*(?:MR|MRS|MS)\s+(?:JUSTICE\s+)?[A-Z][A-Z\s]+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*(?:Vice\s+President|President)\s*$', '', text, flags=re.IGNORECASE)
        
        # Remove case reference lines at the end
        text = re.sub(r'\n\s*[A-Z]+\s+v\s+Secretary of State.*?$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*[A-Z]+/\d+/\d+\s*$', '', text)
        
        # 11. Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots to ellipsis
        text = re.sub(r'-{3,}', '---', text)   # Multiple dashes
        
        # 12. Remove empty lines at start and end, and excessive blank lines
        text = text.strip()
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        
        # 13. Remove very short lines that are likely artifacts (less than 3 characters)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) >= 3 or line == '':  # Keep empty lines for paragraph breaks
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 14. Final cleanup - remove any remaining administrative headers
        text = re.sub(r'^\s*(?:AM|PM)\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Between\s*\n', '', text, flags=re.MULTILINE)
        
        # 15. Ensure text starts with substantive content (paragraph 1 or similar)
        # Skip any remaining header-like content until we reach numbered paragraphs or substantial text
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            line = line.strip()
            # Look for start of substantial content
            if (re.match(r'^\d+\.', line) or  # Numbered paragraph
                len(line) > 50 or  # Long line likely to be substantial content
                any(keyword in line.lower() for keyword in ['appellant', 'applicant', 'claim', 'appeal', 'tribunal'])):
                start_idx = i
                break
        
        if start_idx > 0:
            text = '\n'.join(lines[start_idx:])
        
        return text.strip()
    
    def standardize_date(self, date_str: str) -> Optional[str]:
        """
        Standardize various date formats to YYYY-MM-DD.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Standardized date string or None if parsing fails
        """
        if not date_str or pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Common date patterns in UK legal documents
        patterns = [
            r'(\d{1,2})\s+(\w{3})\s+(\d{4})',  # "31 Oct 2000"
            r'(\d{1,2})/(\d{1,2})/(\d{4})',    # "31/10/2000"
            r'(\d{4})-(\d{1,2})-(\d{1,2})',    # "2000-10-31"
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',    # "31 October 2000"
        ]
        
        month_names = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'oct': 10, 'october': 10,
            'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        
        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    if pattern == patterns[0] or pattern == patterns[3]:  # Month name formats
                        day, month_str, year = match.groups()
                        month = month_names.get(month_str.lower())
                        if month:
                            return f"{year}-{month:02d}-{int(day):02d}"
                    elif pattern == patterns[1]:  # DD/MM/YYYY
                        day, month, year = match.groups()
                        return f"{year}-{int(month):02d}-{int(day):02d}"
                    elif pattern == patterns[2]:  # YYYY-MM-DD (already standard)
                        return date_str
                except ValueError:
                    continue
        
        return None
    
    def extract_case_metadata(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and clean metadata from case JSON.
        
        Args:
            case_data: Raw case data dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        metadata = {}
        
        # Extract standard fields
        for field in self.metadata_fields:
            metadata[field] = case_data.get(field, None)
        
        # Standardize dates
        date_fields = ['promulgation_date', 'hearing_date', 'publication_date', 'last_updated']
        for field in date_fields:
            if field in metadata and metadata[field]:
                metadata[f"{field}_standardized"] = self.standardize_date(metadata[field])
        
        # Extract additional derived features
        metadata['has_pdf_url'] = bool(metadata.get('pdf_url'))
        metadata['has_word_url'] = bool(metadata.get('word_url'))
        
        # Extract year from reference number
        if metadata.get('reference_number'):
            year_match = re.search(r'\[(\d{4})\]', metadata['reference_number'])
            metadata['case_year'] = int(year_match.group(1)) if year_match else None
        
        # Clean text fields
        text_fields = ['case_title', 'appellant_name', 'judges', 'country']
        for field in text_fields:
            if metadata.get(field):
                metadata[field] = str(metadata[field]).strip()
        
        return metadata
    
    def extract_last_section(self, text: str, max_tokens: int = 512) -> str:
        """
        Extract the last section of text for outcome classification.
        Uses approximate token counting (1 token ≈ 4 characters for English text).
        
        Args:
            text: Cleaned decision text
            max_tokens: Maximum number of tokens to extract (default 512 for Legal BERT)
            
        Returns:
            Last section of text within token limit
        """
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Approximate token calculation: 1 token ≈ 4 characters for English
        # Being conservative, use 3.5 characters per token to ensure we don't exceed limit
        max_chars = int(max_tokens * 3.5)
        
        if len(text) <= max_chars:
            # Text is already within limit, return as is
            return text
        
        # Extract the last section
        last_section = text[-max_chars:]
        
        # Try to start from a sentence boundary to avoid cutting mid-sentence
        # Look for the first sentence start in the extracted section
        sentences_start_patterns = [
            r'\.\s+[A-Z]',  # Period followed by space and capital letter
            r'\.\s+\d+\.',  # Period followed by numbered paragraph
            r'\n\d+\.',     # New line with numbered paragraph
            r'\n[A-Z]'      # New line with capital letter
        ]
        
        best_start = 0
        for pattern in sentences_start_patterns:
            matches = list(re.finditer(pattern, last_section))
            if matches:
                # Take the first match to start from a clean sentence boundary
                match_start = matches[0].start() + 1  # +1 to include the period/newline
                if match_start < len(last_section) * 0.3:  # Don't cut too much
                    best_start = match_start
                    break
        
        # If we found a good starting point, use it
        if best_start > 0:
            last_section = last_section[best_start:].strip()
        
        return last_section
    
    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count using approximate conversion.
        Legal BERT typically uses WordPiece tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text or pd.isna(text):
            return 0
        
        # Conservative estimate: 1 token ≈ 3.5 characters for legal text
        return int(len(str(text)) / 3.5)
    
    def process_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Processed case data or None if processing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            
            # Extract metadata
            processed_case = self.extract_case_metadata(case_data)
            
            # Clean decision text
            raw_decision_text = case_data.get('decision_text', '')
            processed_case['decision_text_cleaned'] = self.clean_decision_text(raw_decision_text)
            processed_case['decision_text_length'] = len(processed_case['decision_text_cleaned'])
            processed_case['decision_text_word_count'] = len(processed_case['decision_text_cleaned'].split())
            
            # Extract last section
            processed_case['decision_text_last_section'] = self.extract_last_section(processed_case['decision_text_cleaned'])
            processed_case['last_section_length'] = len(processed_case['decision_text_last_section'])
            processed_case['last_section_token_count'] = self.estimate_token_count(processed_case['decision_text_last_section'])
            
            # Add file metadata
            processed_case['source_file'] = file_path.name
            processed_case['file_size_kb'] = file_path.stat().st_size / 1024
            
            return processed_case
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_all_files(self) -> pd.DataFrame:
        """
        Process all JSON files in the directory.
        
        Returns:
            DataFrame containing all processed cases
        """
        logger.info(f"Starting to process files in {self.json_dir}")
        
        # Find all JSON files
        json_files = list(self.json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.json_dir}")
        
        processed_cases = []
        failed_files = []
        
        # Use tqdm for progress bar
        for file_path in tqdm(json_files, desc="Processing legal cases", unit="files"):
            processed_case = self.process_single_file(file_path)
            if processed_case:
                processed_cases.append(processed_case)
            else:
                failed_files.append(file_path.name)
        
        logger.info(f"Successfully processed {len(processed_cases)} files")
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files[:10]}...")
        
        # Create DataFrame
        df = pd.DataFrame(processed_cases)
        
        # Sort by case year and reference number
        if 'case_year' in df.columns and 'reference_number' in df.columns:
            df = df.sort_values(['case_year', 'reference_number'])
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the processed dataset.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary of summary statistics
        """
        stats = {
            'total_cases': len(df),
            'date_range': {
                'earliest_year': df['case_year'].min() if 'case_year' in df.columns else None,
                'latest_year': df['case_year'].max() if 'case_year' in df.columns else None
            },
            'countries': df['country'].value_counts().to_dict() if 'country' in df.columns else {},
            'text_statistics': {
                'avg_text_length': df['decision_text_length'].mean() if 'decision_text_length' in df.columns else 0,
                'avg_word_count': df['decision_text_word_count'].mean() if 'decision_text_word_count' in df.columns else 0,
                'min_text_length': df['decision_text_length'].min() if 'decision_text_length' in df.columns else 0,
                'max_text_length': df['decision_text_length'].max() if 'decision_text_length' in df.columns else 0,
                'avg_last_section_length': df['last_section_length'].mean() if 'last_section_length' in df.columns else 0,
                'avg_last_section_tokens': df['last_section_token_count'].mean() if 'last_section_token_count' in df.columns else 0,
                'max_last_section_tokens': df['last_section_token_count'].max() if 'last_section_token_count' in df.columns else 0
            },
            'data_quality': {
                'missing_decision_text': df['decision_text_cleaned'].isna().sum() if 'decision_text_cleaned' in df.columns else 0,
                'empty_decision_text': (df['decision_text_cleaned'] == '').sum() if 'decision_text_cleaned' in df.columns else 0,
                'has_pdf_url': df['has_pdf_url'].sum() if 'has_pdf_url' in df.columns else 0,
                'has_word_url': df['has_word_url'].sum() if 'has_word_url' in df.columns else 0
            }
        }
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_legal_cases.parquet"):
        """
        Save processed DataFrame to Parquet with compression, with CSV fallback.
        
        Args:
            df: Processed DataFrame
            filename: Output filename (should end with .parquet)
        """
        output_path = self.output_dir / filename
        
        # Try to save as Parquet first
        try:
            df.to_parquet(output_path, compression='snappy', index=False)
            logger.info(f"Saved processed data to {output_path}")
            parquet_saved = True
        except ImportError as e:
            logger.warning(f"Could not save as Parquet: {e}")
            logger.info("Falling back to CSV format...")
            parquet_saved = False
        
        # Always save a CSV version as backup (and main if parquet failed)
        csv_path = self.output_dir / "processed_legal_cases.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved processed data to {csv_path}")
        
        # Also save a CSV sample for quick inspection
        sample_path = self.output_dir / "sample_processed_cases.csv"
        df.head(100).to_csv(sample_path, index=False, encoding='utf-8')
        logger.info(f"Saved sample (100 cases) to {sample_path}")
        
        # Also save summary statistics
        stats = self.generate_summary_statistics(df)
        stats_path = self.output_dir / "summary_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved summary statistics to {stats_path}")
        
        return output_path if parquet_saved else csv_path

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    start_time = datetime.now()
    
    # Configuration
    json_directory = "data/json"
    output_directory = "preprocessing/processed_data"
    
    # Initialize preprocessor
    preprocessor = LegalCasePreprocessor(json_directory, output_directory)
    
    # Process all files
    logger.info("Starting preprocessing pipeline...")
    logger.info(f"Processing directory: {json_directory}")
    
    df = preprocessor.process_all_files()
    
    # Save results
    output_path = preprocessor.save_processed_data(df)
    
    # Calculate processing time
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    # Display summary
    stats = preprocessor.generate_summary_statistics(df)
    print("\n" + "="*70)
    print("🎉 PREPROCESSING COMPLETE")
    print("="*70)
    print(f"📊 Total cases processed: {stats['total_cases']:,}")
    print(f"⏱️  Processing time: {processing_time}")
    print(f"📅 Date range: {stats['date_range']['earliest_year']} - {stats['date_range']['latest_year']}")
    print(f"📏 Average text length: {stats['text_statistics']['avg_text_length']:,.0f} characters")
    print(f"📝 Average word count: {stats['text_statistics']['avg_word_count']:,.0f} words")
    print(f"🎯 Average last section: {stats['text_statistics']['avg_last_section_tokens']:.0f} tokens")
    print(f"🌍 Top 5 countries: {dict(list(stats['countries'].items())[:5])}")
    print(f"💾 Data saved to: {output_path}")
    print(f"📈 Sample saved to: {output_path.parent / 'sample_processed_cases.csv'}")
    
    # File size information
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"📦 Output file size: {file_size_mb:.1f} MB")
    
    print("="*70)
    print("🚀 Ready for Legal BERT classification!")
    print("💡 Load with: pd.read_parquet('{}')".format(output_path))
    print("="*70)
    
    return df

if __name__ == "__main__":
    df = main() 