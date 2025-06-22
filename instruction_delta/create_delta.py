#!/usr/bin/env python3
"""
create_delta.py - Compute instruction residual between base and instruct models

This tool computes the delta (Δ = INSTRUCT - BASE) between a base model and its
instruction-tuned version, saving the result as a compact delta file that can
be applied to other models fine-tuned from the same base.

Usage:
    python create_delta.py \
        --base_path meta-llama/Meta-Llama-3-8B \
        --instruct_path meta-llama/Meta-Llama-3-8B-Instruct \
        --delta_out ./deltas/llama3_8b_instruction_delta \
        --dtype fp16

The output will be:
    ./deltas/llama3_8b_instruction_delta/delta.safetensors
    ./deltas/llama3_8b_instruction_delta/delta_meta.json
"""

import argparse
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_model_files(repo_or_path: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Download model files from HuggingFace Hub or return local paths.
    
    Returns:
        Dict mapping filename to local path
    """
    if os.path.exists(repo_or_path):
        # Local directory
        model_dir = Path(repo_or_path)
        files = {}
        
        # Find all safetensors files
        for file_path in model_dir.glob("*.safetensors"):
            files[file_path.name] = str(file_path)
            
        if not files:
            raise ValueError(f"No .safetensors files found in {repo_or_path}")
            
        return files
    else:
        # HuggingFace Hub repository
        try:
            repo_files = list_repo_files(repo_or_path)
            safetensors_files = [f for f in repo_files if f.endswith('.safetensors')]
            
            if not safetensors_files:
                raise ValueError(f"No .safetensors files found in {repo_or_path}")
            
            files = {}
            logger.info(f"Downloading {len(safetensors_files)} files from {repo_or_path}")
            
            for filename in tqdm(safetensors_files, desc="Downloading"):
                local_path = hf_hub_download(
                    repo_id=repo_or_path,
                    filename=filename,
                    cache_dir=cache_dir
                )
                files[filename] = local_path
                
            return files
            
        except Exception as e:
            raise ValueError(f"Could not access {repo_or_path}: {e}")


def load_model_weights(file_paths: Dict[str, str], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load model weights from safetensors files."""
    weights = {}
    
    logger.info(f"Loading weights from {len(file_paths)} files")
    
    for filename, filepath in tqdm(file_paths.items(), desc="Loading weights"):
        file_weights = load_file(filepath, device="cpu")
        
        # Convert to specified dtype, but keep embeddings and layer norms in fp32
        for key, tensor in file_weights.items():
            if dtype == torch.float16 and ('embed' in key or 'norm' in key or 'ln' in key):
                # Keep critical layers in fp32
                weights[key] = tensor.float()
            else:
                weights[key] = tensor.to(dtype)
                
    logger.info(f"Loaded {len(weights)} tensors")
    return weights


def compute_delta(
    base_weights: Dict[str, torch.Tensor],
    instruct_weights: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Compute delta = instruct - base element-wise."""
    
    # Verify tensor keys match
    base_keys = set(base_weights.keys())
    instruct_keys = set(instruct_weights.keys())
    
    if base_keys != instruct_keys:
        missing_in_base = instruct_keys - base_keys
        missing_in_instruct = base_keys - instruct_keys
        
        error_msg = "Tensor keys don't match between base and instruct models:\n"
        if missing_in_base:
            error_msg += f"Missing in base: {missing_in_base}\n"
        if missing_in_instruct:
            error_msg += f"Missing in instruct: {missing_in_instruct}\n"
            
        raise ValueError(error_msg)
    
    delta = {}
    total_params = 0
    changed_params = 0
    
    logger.info("Computing delta (instruct - base)")
    
    for key in tqdm(base_keys, desc="Computing delta"):
        base_tensor = base_weights[key]
        instruct_tensor = instruct_weights[key]
        
        # Verify shapes match
        if base_tensor.shape != instruct_tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: {base_tensor.shape} vs {instruct_tensor.shape}")
        
        # Compute delta
        delta_tensor = instruct_tensor - base_tensor
        delta[key] = delta_tensor
        
        # Track statistics
        total_params += delta_tensor.numel()
        changed_params += (delta_tensor != 0).sum().item()
    
    logger.info(f"Delta computed: {changed_params:,}/{total_params:,} parameters changed ({100*changed_params/total_params:.2f}%)")
    
    return delta


def save_delta(
    delta: Dict[str, torch.Tensor],
    output_dir: str,
    base_path: str,
    instruct_path: str,
    base_files: Dict[str, str],
    instruct_files: Dict[str, str],
    dtype: torch.dtype
):
    """Save delta weights and metadata."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save delta weights
    delta_file = output_path / "delta.safetensors"
    logger.info(f"Saving delta to {delta_file}")
    save_file(delta, str(delta_file))
    
    # Compute file hashes for integrity checking
    base_hashes = {}
    for filename, filepath in base_files.items():
        base_hashes[filename] = compute_sha256(filepath)
        
    instruct_hashes = {}
    for filename, filepath in instruct_files.items():
        instruct_hashes[filename] = compute_sha256(filepath)
    
    # Compute delta statistics
    total_params = sum(t.numel() for t in delta.values())
    l2_norm = torch.sqrt(sum((t ** 2).sum() for t in delta.values())).item()
    
    # Create metadata
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "base_model": base_path,
        "instruct_model": instruct_path,
        "dtype": str(dtype).split('.')[-1],  # e.g., "float16"
        "total_parameters": total_params,
        "l2_norm": l2_norm,
        "base_file_hashes": base_hashes,
        "instruct_file_hashes": instruct_hashes,
        "tensor_keys": list(delta.keys()),
        "tensor_shapes": {k: list(v.shape) for k, v in delta.items()}
    }
    
    # Save metadata
    meta_file = output_path / "delta_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Delta saved to {output_dir}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"L2 norm of delta: {l2_norm:.6f}")
    
    return delta_file, meta_file


def main():
    parser = argparse.ArgumentParser(
        description="Compute instruction residual between base and instruct models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--base_path",
        required=True,
        help="Path or HuggingFace repo ID for base model (e.g., meta-llama/Meta-Llama-3-8B)"
    )
    
    parser.add_argument(
        "--instruct_path", 
        required=True,
        help="Path or HuggingFace repo ID for instruct model (e.g., meta-llama/Meta-Llama-3-8B-Instruct)"
    )
    
    parser.add_argument(
        "--delta_out",
        required=True,
        help="Output directory for delta files"
    )
    
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Data type for delta computation (default: fp16, keeps embeddings/norms in fp32)"
    )
    
    parser.add_argument(
        "--cache_dir",
        help="Cache directory for HuggingFace downloads"
    )
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    
    try:
        logger.info("Starting delta computation")
        logger.info(f"Base model: {args.base_path}")
        logger.info(f"Instruct model: {args.instruct_path}")
        logger.info(f"Output: {args.delta_out}")
        logger.info(f"Dtype: {args.dtype}")
        
        # Download/locate model files
        logger.info("Locating base model files...")
        base_files = download_model_files(args.base_path, args.cache_dir)
        
        logger.info("Locating instruct model files...")
        instruct_files = download_model_files(args.instruct_path, args.cache_dir)
        
        # Load weights
        logger.info("Loading base model weights...")
        base_weights = load_model_weights(base_files, dtype)
        
        logger.info("Loading instruct model weights...")
        instruct_weights = load_model_weights(instruct_files, dtype)
        
        # Compute delta
        delta = compute_delta(base_weights, instruct_weights)
        
        # Save delta
        delta_file, meta_file = save_delta(
            delta, args.delta_out, args.base_path, args.instruct_path,
            base_files, instruct_files, dtype
        )
        
        logger.info("✅ Delta computation completed successfully!")
        logger.info(f"📁 Delta file: {delta_file}")
        logger.info(f"📄 Metadata: {meta_file}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main() 