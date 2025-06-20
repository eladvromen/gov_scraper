import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_post_brexit_data(input_dir: str, output_dir: str, start_year: int, end_year: int):
    """
    Filter post-Brexit data for a specific year range and create a new dataset.
    Optimized version using pandas and numpy for faster processing.
    
    Args:
        input_dir: Directory containing the original post-Brexit dataset
        output_dir: Directory to save the filtered dataset
        start_year: Start year (inclusive)
        end_year: End year (inclusive)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read and filter metadata
    logger.info(f"Reading and filtering metadata for years {start_year}-{end_year}...")
    metadata_df = pd.read_csv(input_path / "post_brexit_2018_2025_metadata.csv")
    filtered_df = metadata_df[
        (metadata_df['original_case_year'] >= start_year) & 
        (metadata_df['original_case_year'] <= end_year)
    ].copy()
    
    # Get cases to keep
    cases_to_keep = set(filtered_df['case_id'].unique())
    logger.info(f"Found {len(cases_to_keep)} unique cases from {start_year}-{end_year}")
    
    # Process chunks file using pandas
    logger.info("Processing chunks file...")
    chunks_df = pd.read_json(input_path / "post_brexit_2018_2025_chunks.jsonl", lines=True)
    filtered_chunks = chunks_df[chunks_df['case_id'].isin(cases_to_keep)]
    
    # Save filtered chunks
    filtered_chunks.to_json(output_path / f"post_brexit_{start_year}_{end_year}_chunks.jsonl", 
                          orient='records', lines=True)
    
    # Save filtered metadata
    logger.info("Saving filtered metadata...")
    filtered_df.to_csv(output_path / f"post_brexit_{start_year}_{end_year}_metadata.csv", index=False)
    
    # Generate statistics
    stats = {
        "dataset_name": f"post_brexit_{start_year}_{end_year}",
        "total_chunks": int(len(filtered_chunks)),
        "total_tokens": int(filtered_df['token_count'].sum()),
        "avg_tokens_per_chunk": float(filtered_df['token_count'].mean()),
        "min_tokens": int(filtered_df['token_count'].min()),
        "max_tokens": int(filtered_df['token_count'].max()),
        "total_characters": int(filtered_df['chunk_length'].sum()),
        "unique_cases": int(len(cases_to_keep))
    }
    
    # Save statistics
    with open(output_path / f"post_brexit_{start_year}_{end_year}_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Process training text
    logger.info("Processing training text file...")
    training_chunks = []
    current_chunk = []
    current_case = None
    
    # Read training text in chunks for memory efficiency
    chunk_size = 100000  # Adjust based on available memory
    with open(input_path / "post_brexit_2018_2025_training.txt", 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing training text"):
            if line.startswith("###"):
                # If we have a chunk and it's from a case we want to keep
                if current_chunk and current_case in cases_to_keep:
                    training_chunks.append(''.join(current_chunk))
                # Start new chunk
                current_case = line.strip().split()[1]  # Get case ID
                current_chunk = [line]
            else:
                current_chunk.append(line)
        # Don't forget the last chunk
        if current_chunk and current_case in cases_to_keep:
            training_chunks.append(''.join(current_chunk))
    
    # Write filtered training text
    with open(output_path / f"post_brexit_{start_year}_{end_year}_training.txt", 'w', encoding='utf-8') as f:
        for chunk in training_chunks:
            f.write(chunk)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("FILTERING COMPLETE")
    logger.info("="*50)
    logger.info(f"Original cases: {len(metadata_df['case_id'].unique())}")
    logger.info(f"Filtered cases ({start_year}-{end_year}): {len(cases_to_keep)}")
    logger.info(f"Original chunks: {len(metadata_df)}")
    logger.info(f"Filtered chunks: {len(filtered_chunks)}")
    logger.info(f"Total tokens in filtered dataset: {stats['total_tokens']:,}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("="*50)
    
    return stats

if __name__ == "__main__":
    input_directory = "preprocessing/outputs/llama_training_ready/post_brexit_2018_2025"
    
    # Create 2019-2025 dataset
    output_directory_2019_2025 = "preprocessing/outputs/llama_training_ready/post_brexit_2019_2025"
    stats_2019_2025 = filter_post_brexit_data(input_directory, output_directory_2019_2025, 2019, 2025)
    
    # Create 2018-2022 dataset
    output_directory_2018_2022 = "preprocessing/outputs/llama_training_ready/post_brexit_2018_2022"
    stats_2018_2022 = filter_post_brexit_data(input_directory, output_directory_2018_2022, 2018, 2022)
    
    # Create 2020-2025 dataset
    output_directory_2020_2025 = "preprocessing/outputs/llama_training_ready/post_brexit_2020_2025"
    stats_2020_2025 = filter_post_brexit_data(input_directory, output_directory_2020_2025, 2020, 2025)
    
    # Create 2017-2019 dataset
    output_directory_2017_2019 = "preprocessing/outputs/llama_training_ready/post_brexit_2017_2019"
    stats_2017_2019 = filter_post_brexit_data(input_directory, output_directory_2017_2019, 2017, 2019) 