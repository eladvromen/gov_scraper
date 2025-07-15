#!/usr/bin/env python3
"""
Generic Vector Utilities for Divergence Analysis
Shared utilities for vector manipulation, weighting, and preprocessing
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

class VectorProcessor:
    """Generic vector processing and manipulation utilities"""
    
    def __init__(self, config: Dict):
        """Initialize with configuration settings"""
        self.config = config
        self.weighting_config = config.get('divergence_analysis', {}).get('weighting', {})
        
    def extract_vectors(self, df: pd.DataFrame, 
                       magnitude_cols: List[str], 
                       significance_cols: Optional[List[str]] = None,
                       group_size_cols: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Extract vectors from dataframe with optional significance and weighting
        
        Args:
            df: Source dataframe
            magnitude_cols: Column names for magnitude values
            significance_cols: Column names for significance indicators
            group_size_cols: Column names for group sizes (for weighting)
            
        Returns:
            Dictionary containing extracted vectors
        """
        vectors = {}
        
        # Extract magnitude vectors
        for i, col in enumerate(magnitude_cols):
            if col in df.columns:
                vector = df[col].fillna(0).values  # Fill NaN with 0 for missing data
                vectors[f'magnitude_{i}'] = vector
                logger.info(f"Extracted magnitude vector from {col}: {len(vector)} dimensions")
            else:
                logger.warning(f"Column {col} not found in dataframe")
        
        # Extract significance vectors if provided
        if significance_cols:
            for i, col in enumerate(significance_cols):
                if col in df.columns:
                    vector = df[col].fillna(False).astype(bool).values
                    vectors[f'significance_{i}'] = vector
                    logger.info(f"Extracted significance vector from {col}: {np.sum(vector)} significant out of {len(vector)}")
                else:
                    logger.warning(f"Significance column {col} not found in dataframe")
        
        # Extract group size vectors for weighting if provided
        if group_size_cols:
            for i, col in enumerate(group_size_cols):
                if col in df.columns:
                    vector = df[col].fillna(1).values  # Fill NaN with 1 (equal weight)
                    vectors[f'group_size_{i}'] = vector
                    logger.info(f"Extracted group size vector from {col}: min={np.min(vector)}, max={np.max(vector)}")
                else:
                    logger.warning(f"Group size column {col} not found in dataframe")
        
        return vectors
    
    def create_weights(self, vectors: Dict[str, np.ndarray], 
                      strategy: str = "min_sample_size") -> np.ndarray:
        """
        Create weight vector based on strategy
        
        Args:
            vectors: Dictionary containing vectors (must include group_size vectors for weighting)
            strategy: Weighting strategy ("min_sample_size", "equal", "log_sample_size")
            
        Returns:
            Weight vector
        """
        if not self.weighting_config.get('enabled', False):
            # Return equal weights
            vector_length = len(list(vectors.values())[0])
            return np.ones(vector_length)
        
        # Find group size vectors
        group_size_vectors = [v for k, v in vectors.items() if k.startswith('group_size_')]
        
        if not group_size_vectors:
            logger.warning("No group size vectors found, using equal weights")
            vector_length = len(list(vectors.values())[0])
            return np.ones(vector_length)
        
        if strategy == "min_sample_size":
            # Use minimum sample size across models as weight
            weights = np.minimum.reduce(group_size_vectors)
        elif strategy == "equal":
            weights = np.ones(len(group_size_vectors[0]))
        elif strategy == "log_sample_size":
            # Use log of minimum sample size
            min_sizes = np.minimum.reduce(group_size_vectors)
            weights = np.log1p(min_sizes)  # log(1 + x) to avoid log(0)
        else:
            logger.warning(f"Unknown weighting strategy {strategy}, using equal weights")
            weights = np.ones(len(group_size_vectors[0]))
        
        # Apply minimum weight threshold
        min_weight = self.weighting_config.get('min_weight', 1)
        weights = np.maximum(weights, min_weight)
        
        logger.info(f"Created weights using strategy '{strategy}': min={np.min(weights):.2f}, max={np.max(weights):.2f}, mean={np.mean(weights):.2f}")
        
        return weights
    
    def validate_vectors(self, vectors: Dict[str, np.ndarray]) -> bool:
        """
        Validate that all vectors have same length and are valid
        
        Args:
            vectors: Dictionary of vectors to validate
            
        Returns:
            True if all vectors are valid
        """
        if not vectors:
            logger.error("No vectors provided for validation")
            return False
        
        # Check all vectors have same length
        lengths = [len(v) for v in vectors.values()]
        if len(set(lengths)) > 1:
            logger.error(f"Vector length mismatch: {dict(zip(vectors.keys(), lengths))}")
            return False
        
        # Check for empty vectors
        if lengths[0] == 0:
            logger.error("Empty vectors provided")
            return False
        
        # Check for all-NaN vectors
        for name, vector in vectors.items():
            if np.all(np.isnan(vector)):
                logger.error(f"Vector '{name}' contains all NaN values")
                return False
        
        logger.info(f"Vector validation passed: {len(vectors)} vectors, {lengths[0]} dimensions each")
        return True
    
    def clean_vectors(self, vectors: Dict[str, np.ndarray], 
                     remove_nan_rows: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Clean vectors by removing NaN values and invalid entries
        
        Args:
            vectors: Dictionary of vectors to clean
            remove_nan_rows: Whether to remove rows with any NaN values
            
        Returns:
            Tuple of (cleaned_vectors, valid_indices)
        """
        if not vectors:
            return vectors, np.array([])
        
        vector_length = len(list(vectors.values())[0])
        valid_mask = np.ones(vector_length, dtype=bool)
        
        if remove_nan_rows:
            # Find rows with any NaN values
            for name, vector in vectors.items():
                if np.issubdtype(vector.dtype, np.number):  # Only check numeric vectors for NaN
                    nan_mask = np.isnan(vector)
                    valid_mask &= ~nan_mask
                    if np.any(nan_mask):
                        logger.info(f"Found {np.sum(nan_mask)} NaN values in vector '{name}'")
        
        # Apply mask to all vectors
        cleaned_vectors = {}
        for name, vector in vectors.items():
            cleaned_vectors[name] = vector[valid_mask]
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < vector_length:
            logger.info(f"Cleaned vectors: removed {vector_length - len(valid_indices)} invalid rows, {len(valid_indices)} rows remaining")
        
        return cleaned_vectors, valid_indices

def calculate_divergence_metrics(vector1: np.ndarray, 
                                vector2: np.ndarray,
                                weights: Optional[np.ndarray] = None,
                                methods: List[str] = None) -> Dict[str, float]:
    """
    Calculate various divergence metrics between two vectors
    
    Args:
        vector1: First vector
        vector2: Second vector
        weights: Optional weight vector for weighted calculations
        methods: List of methods to calculate
        
    Returns:
        Dictionary of divergence metrics
    """
    if methods is None:
        methods = ['cosine_similarity', 'pearson_correlation', 'euclidean_distance', 
                  'manhattan_distance', 'mean_absolute_diff']
    
    metrics = {}
    
    # Handle weights
    if weights is not None:
        # Normalize weights
        weights = weights / np.sum(weights) * len(weights)
    
    try:
        # Cosine similarity
        if 'cosine_similarity' in methods:
            cos_sim = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
            metrics['cosine_similarity'] = float(cos_sim)
            metrics['cosine_distance'] = float(1 - cos_sim)
        
        # Pearson correlation
        if 'pearson_correlation' in methods:
            if weights is not None:
                # Weighted correlation (approximate)
                mean1 = np.average(vector1, weights=weights)
                mean2 = np.average(vector2, weights=weights)
                cov = np.average((vector1 - mean1) * (vector2 - mean2), weights=weights)
                var1 = np.average((vector1 - mean1)**2, weights=weights)
                var2 = np.average((vector2 - mean2)**2, weights=weights)
                if var1 > 0 and var2 > 0:
                    corr = cov / np.sqrt(var1 * var2)
                else:
                    corr = 0.0
                metrics['pearson_correlation'] = float(corr)
            else:
                corr, p_val = pearsonr(vector1, vector2)
                metrics['pearson_correlation'] = float(corr) if not np.isnan(corr) else 0.0
                metrics['pearson_p_value'] = float(p_val) if not np.isnan(p_val) else 1.0
        
        # Spearman correlation
        if 'spearman_correlation' in methods:
            if weights is None:  # Spearman doesn't work well with weights
                corr, p_val = spearmanr(vector1, vector2)
                metrics['spearman_correlation'] = float(corr) if not np.isnan(corr) else 0.0
                metrics['spearman_p_value'] = float(p_val) if not np.isnan(p_val) else 1.0
        
        # Distance metrics
        diff = vector1 - vector2
        
        if 'euclidean_distance' in methods:
            if weights is not None:
                euclidean_dist = np.sqrt(np.sum(weights * diff**2) / np.sum(weights))
            else:
                euclidean_dist = np.linalg.norm(diff)
            metrics['euclidean_distance'] = float(euclidean_dist)
        
        if 'manhattan_distance' in methods:
            if weights is not None:
                manhattan_dist = np.sum(weights * np.abs(diff)) / np.sum(weights)
            else:
                manhattan_dist = np.sum(np.abs(diff))
            metrics['manhattan_distance'] = float(manhattan_dist)
        
        if 'mean_absolute_diff' in methods:
            if weights is not None:
                mean_abs_diff = np.average(np.abs(diff), weights=weights)
            else:
                mean_abs_diff = np.mean(np.abs(diff))
            metrics['mean_absolute_difference'] = float(mean_abs_diff)
        
        # Additional metrics
        metrics['vector_length'] = len(vector1)
        if weights is not None:
            metrics['effective_sample_size'] = float(np.sum(weights))
        
    except Exception as e:
        logger.error(f"Error calculating divergence metrics: {str(e)}")
        # Return default values
        for method in methods:
            if method not in metrics:
                metrics[method] = 0.0
    
    return metrics

def create_comparison_labels(df: pd.DataFrame, 
                           id_columns: List[str]) -> List[str]:
    """
    Create human-readable labels for vector comparisons
    
    Args:
        df: Source dataframe
        id_columns: Columns to use for creating labels
        
    Returns:
        List of comparison labels
    """
    labels = []
    for _, row in df.iterrows():
        label_parts = []
        for col in id_columns:
            if col in df.columns and pd.notna(row[col]):
                label_parts.append(str(row[col]))
        
        if label_parts:
            labels.append(" | ".join(label_parts))
        else:
            labels.append(f"Comparison_{len(labels)}")
    
    return labels 