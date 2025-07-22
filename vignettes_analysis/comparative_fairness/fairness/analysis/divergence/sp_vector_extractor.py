#!/usr/bin/env python3
"""
Statistical Parity Vector Extractor
Fairness-specific functionality for extracting SP vectors from unified dataframes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.vector_utils import VectorProcessor, create_comparison_labels

logger = logging.getLogger(__name__)

class SPVectorExtractor:
    """Statistical Parity vector extraction for fairness analysis"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        self.fairness_config = config.get('vector_extraction', {}).get('fairness', {})
        self.vector_processor = VectorProcessor(config)
        
    def load_fairness_dataframe(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load the unified fairness dataframe"""
        
        if file_path is None:
            file_path = self.config.get('data_sources', {}).get('fairness_dataframe')
        
        if not file_path:
            raise ValueError("No fairness dataframe path provided in configuration")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fairness dataframe not found: {file_path}")
        
        logger.info(f"Loading fairness dataframe from {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Loaded fairness dataframe: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Protected attributes: {df['protected_attribute'].nunique()}")
        logger.info(f"Topics: {df['topic'].nunique()}")
        
        return df
    
    def extract_sp_magnitude_vectors(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract statistical parity magnitude vectors for both models
        
        Args:
            df: Unified fairness dataframe
            
        Returns:
            Dictionary containing SP magnitude vectors
        """
        logger.info("Extracting SP magnitude vectors")
        
        # Define column mappings
        pre_brexit_col = 'pre_brexit_model_statistical_parity'
        post_brexit_col = 'post_brexit_model_statistical_parity'
        
        # Validate required columns exist
        required_cols = [pre_brexit_col, post_brexit_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract magnitude vectors
        vectors = {}
        
        # Pre-Brexit SP vector
        pre_brexit_vector = df[pre_brexit_col].fillna(0).values
        vectors['pre_brexit_sp_magnitude'] = pre_brexit_vector
        
        # Post-Brexit SP vector
        post_brexit_vector = df[post_brexit_col].fillna(0).values
        vectors['post_brexit_sp_magnitude'] = post_brexit_vector
        
        # Calculate difference vector
        difference_vector = post_brexit_vector - pre_brexit_vector
        vectors['sp_magnitude_difference'] = difference_vector
        
        logger.info(f"Extracted SP magnitude vectors: {len(pre_brexit_vector)} dimensions")
        logger.info(f"Pre-Brexit SP: mean={np.mean(pre_brexit_vector):.4f}, std={np.std(pre_brexit_vector):.4f}")
        logger.info(f"Post-Brexit SP: mean={np.mean(post_brexit_vector):.4f}, std={np.std(post_brexit_vector):.4f}")
        logger.info(f"SP Difference: mean={np.mean(difference_vector):.4f}, std={np.std(difference_vector):.4f}")
        
        return vectors
    
    def extract_sp_significance_vectors(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract statistical parity significance vectors for both models
        
        Args:
            df: Unified fairness dataframe
            
        Returns:
            Dictionary containing SP significance vectors
        """
        logger.info("Extracting SP significance vectors")
        
        # Define column mappings
        pre_brexit_sig_col = 'pre_brexit_sp_significance'
        post_brexit_sig_col = 'post_brexit_sp_significance'
        
        # Validate required columns exist
        required_cols = [pre_brexit_sig_col, post_brexit_sig_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract significance vectors
        vectors = {}
        
        # Pre-Brexit significance vector
        pre_brexit_sig = df[pre_brexit_sig_col].fillna(False).astype(bool).values
        vectors['pre_brexit_sp_significance'] = pre_brexit_sig
        
        # Post-Brexit significance vector
        post_brexit_sig = df[post_brexit_sig_col].fillna(False).astype(bool).values
        vectors['post_brexit_sp_significance'] = post_brexit_sig
        
        # Calculate significance change vector
        significance_change = pre_brexit_sig != post_brexit_sig
        vectors['sp_significance_change'] = significance_change
        
        # Additional significance patterns
        both_significant = pre_brexit_sig & post_brexit_sig
        vectors['sp_both_significant'] = both_significant
        
        neither_significant = ~pre_brexit_sig & ~post_brexit_sig
        vectors['sp_neither_significant'] = neither_significant
        
        gained_significance = ~pre_brexit_sig & post_brexit_sig
        vectors['sp_gained_significance'] = gained_significance
        
        lost_significance = pre_brexit_sig & ~post_brexit_sig
        vectors['sp_lost_significance'] = lost_significance
        
        # Log statistics
        logger.info(f"Extracted SP significance vectors: {len(pre_brexit_sig)} dimensions")
        logger.info(f"Pre-Brexit significant: {np.sum(pre_brexit_sig)} ({np.mean(pre_brexit_sig):.2%})")
        logger.info(f"Post-Brexit significant: {np.sum(post_brexit_sig)} ({np.mean(post_brexit_sig):.2%})")
        logger.info(f"Significance changes: {np.sum(significance_change)} ({np.mean(significance_change):.2%})")
        logger.info(f"Gained significance: {np.sum(gained_significance)}")
        logger.info(f"Lost significance: {np.sum(lost_significance)}")
        
        return vectors
    
    def extract_group_size_vectors(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Extract group size vectors for weighting
        
        Args:
            df: Unified fairness dataframe
            
        Returns:
            Dictionary containing group size vectors
        """
        logger.info("Extracting group size vectors")
        
        # Define column mappings
        pre_group_col = 'pre_brexit_group_size'
        post_group_col = 'post_brexit_group_size'
        pre_ref_col = 'pre_brexit_reference_size'
        post_ref_col = 'post_brexit_reference_size'
        
        vectors = {}
        
        # Extract group sizes if available
        if pre_group_col in df.columns:
            vectors['pre_brexit_group_size'] = df[pre_group_col].fillna(1).values
        
        if post_group_col in df.columns:
            vectors['post_brexit_group_size'] = df[post_group_col].fillna(1).values
        
        if pre_ref_col in df.columns:
            vectors['pre_brexit_reference_size'] = df[pre_ref_col].fillna(1).values
        
        if post_ref_col in df.columns:
            vectors['post_brexit_reference_size'] = df[post_ref_col].fillna(1).values
        
        # Calculate minimum sample sizes for weighting
        if pre_group_col in df.columns and post_group_col in df.columns:
            min_group_sizes = np.minimum(
                df[pre_group_col].fillna(1).values,
                df[post_group_col].fillna(1).values
            )
            vectors['min_group_size'] = min_group_sizes
        
        if pre_ref_col in df.columns and post_ref_col in df.columns:
            min_ref_sizes = np.minimum(
                df[pre_ref_col].fillna(1).values,
                df[post_ref_col].fillna(1).values
            )
            vectors['min_reference_size'] = min_ref_sizes
        
        # Overall minimum sample size for robust weighting
        if 'min_group_size' in vectors and 'min_reference_size' in vectors:
            overall_min_size = np.minimum(vectors['min_group_size'], vectors['min_reference_size'])
            vectors['min_overall_size'] = overall_min_size
            
            logger.info(f"Overall sample sizes: min={np.min(overall_min_size)}, max={np.max(overall_min_size)}, mean={np.mean(overall_min_size):.1f}")
        
        logger.info(f"Extracted {len(vectors)} group size vectors")
        
        return vectors
    
    def create_comparison_labels(self, df: pd.DataFrame) -> List[str]:
        """
        Create human-readable labels for each comparison
        
        Args:
            df: Unified fairness dataframe
            
        Returns:
            List of comparison labels
        """
        logger.info("Creating comparison labels")
        
        labels = []
        for _, row in df.iterrows():
            # Create label from group comparison, protected attribute, and topic
            parts = []
            
            if 'group_comparison' in df.columns and pd.notna(row['group_comparison']):
                parts.append(str(row['group_comparison']))
            
            if 'protected_attribute' in df.columns and pd.notna(row['protected_attribute']):
                parts.append(f"({row['protected_attribute']})")
            
            if 'topic' in df.columns and pd.notna(row['topic']):
                topic = str(row['topic'])
                if len(topic) > 30:  # Truncate long topic names
                    topic = topic[:27] + "..."
                parts.append(f"[{topic}]")
            
            if parts:
                labels.append(" ".join(parts))
            else:
                labels.append(f"Comparison_{len(labels)}")
        
        logger.info(f"Created {len(labels)} comparison labels")
        
        return labels
    
    def extract_all_vectors(self, df: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Extract all vectors (magnitude, significance, group sizes) at once
        
        Args:
            df: Optional dataframe (will load from config if not provided)
            
        Returns:
            Dictionary containing all extracted vectors
        """
        if df is None:
            df = self.load_fairness_dataframe()
        
        logger.info("Extracting all SP vectors")
        
        # Extract all vector types
        all_vectors = {}
        
        # SP magnitude vectors
        magnitude_vectors = self.extract_sp_magnitude_vectors(df)
        all_vectors.update(magnitude_vectors)
        
        # SP significance vectors
        significance_vectors = self.extract_sp_significance_vectors(df)
        all_vectors.update(significance_vectors)
        
        # Group size vectors
        group_size_vectors = self.extract_group_size_vectors(df)
        all_vectors.update(group_size_vectors)
        
        # Comparison labels
        labels = self.create_comparison_labels(df)
        
        # Validate all vectors have same length
        vector_lengths = [len(v) for v in all_vectors.values()]
        if len(set(vector_lengths)) > 1:
            logger.error(f"Vector length mismatch: {dict(zip(all_vectors.keys(), vector_lengths))}")
            raise ValueError("All vectors must have the same length")
        
        logger.info(f"Successfully extracted {len(all_vectors)} vectors with {vector_lengths[0]} dimensions each")
        
        # Return vectors with labels
        result = {
            'vectors': all_vectors,
            'labels': labels,
            'dataframe_info': {
                'total_comparisons': len(df),
                'protected_attributes': df['protected_attribute'].nunique() if 'protected_attribute' in df.columns else 0,
                'topics': df['topic'].nunique() if 'topic' in df.columns else 0,
                'attributes_breakdown': df['protected_attribute'].value_counts().to_dict() if 'protected_attribute' in df.columns else {}
            }
        }
        
        return result
    
    def get_stratified_vectors(self, df: Optional[pd.DataFrame] = None, 
                              stratify_by: str = 'protected_attribute') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract vectors stratified by a specific column
        
        Args:
            df: Optional dataframe (will load from config if not provided)
            stratify_by: Column to stratify by ('protected_attribute', 'topic', etc.)
            
        Returns:
            Dictionary mapping stratification values to vector dictionaries
        """
        if df is None:
            df = self.load_fairness_dataframe()
        
        if stratify_by not in df.columns:
            raise ValueError(f"Stratification column '{stratify_by}' not found in dataframe")
        
        logger.info(f"Extracting stratified vectors by '{stratify_by}'")
        
        stratified_results = {}
        
        # Get unique values for stratification
        unique_values = df[stratify_by].unique()
        
        for value in unique_values:
            if pd.isna(value):
                continue
                
            # Filter dataframe for this stratification value
            subset_df = df[df[stratify_by] == value].copy()
            
            if len(subset_df) == 0:
                continue
            
            logger.info(f"Processing {stratify_by}='{value}': {len(subset_df)} comparisons")
            
            # Extract vectors for this subset
            try:
                magnitude_vectors = self.extract_sp_magnitude_vectors(subset_df)
                significance_vectors = self.extract_sp_significance_vectors(subset_df)
                group_size_vectors = self.extract_group_size_vectors(subset_df)
                labels = self.create_comparison_labels(subset_df)
                
                subset_vectors = {}
                subset_vectors.update(magnitude_vectors)
                subset_vectors.update(significance_vectors)
                subset_vectors.update(group_size_vectors)
                
                stratified_results[str(value)] = {
                    'vectors': subset_vectors,
                    'labels': labels,
                    'subset_info': {
                        'comparisons': len(subset_df),
                        'stratify_by': stratify_by,
                        'stratify_value': str(value)
                    }
                }
                
            except Exception as e:
                logger.warning(f"Failed to process {stratify_by}='{value}': {str(e)}")
                continue
        
        logger.info(f"Successfully extracted stratified vectors for {len(stratified_results)} values")
        
        return stratified_results 