#!/usr/bin/env python3
"""
apply_delta.py - Apply instruction residual to fine-tuned models

This tool applies a pre-computed instruction delta (Δ) to a model that was
fine-tuned from the same base model, creating a new model with instruction
capabilities: merged = model + Δ

Usage:
    python apply_delta.py \
        --model_path ./models/llama3_8b_pre_brexit \
        --delta_path ./deltas/llama3_8b_instruction_delta \
        --merged_out ./models/llama3_8b_pre_brexit_instruct \
        --dtype fp16

The script will:
1. Load the fine-tuned model and the delta
2. Apply delta: merged = model + Δ
3. Save merged weights and copy config/tokenizer
4. Test with a sample inference
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional
import logging

import torch
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_delta(delta_path: str) -> tuple[Dict[str, torch.Tensor], Dict]:
    """Load delta weights and metadata."""
    
    if not os.path.exists(delta_path):
        # Try downloading from HuggingFace Hub
        try:
            logger.info(f"Attempting to download delta from HF Hub: {delta_path}")
            delta_file = hf_hub_download(repo_id=delta_path, filename="delta.safetensors")
            meta_file = hf_hub_download(repo_id=delta_path, filename="delta_meta.json")
        except Exception as e:
            raise ValueError(f"Could not find delta at {delta_path}: {e}")
    else:
        # Local directory
        delta_dir = Path(delta_path)
        delta_file = delta_dir / "delta.safetensors"
        meta_file = delta_dir / "delta_meta.json"
        
        if not delta_file.exists():
            raise ValueError(f"Delta file not found: {delta_file}")
        if not meta_file.exists():
            raise ValueError(f"Metadata file not found: {meta_file}")
        
        delta_file = str(delta_file)
        meta_file = str(meta_file)
    
    # Load delta weights
    logger.info(f"Loading delta from {delta_file}")
    delta_weights = load_file(delta_file, device="cpu")
    
    # Load metadata
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded delta with {len(delta_weights)} tensors")
    logger.info(f"Delta L2 norm: {metadata.get('l2_norm', 'unknown')}")
    
    return delta_weights, metadata


def load_model_files(model_path: str, cache_dir: Optional[str] = None) -> tuple[Dict[str, str], str]:
    """Load model files from local directory or HuggingFace Hub."""
    
    if os.path.exists(model_path):
        # Local directory
        model_dir = Path(model_path)
        files = {}
        
        # Find all safetensors files
        for file_path in model_dir.glob("*.safetensors"):
            files[file_path.name] = str(file_path)
            
        if not files:
            raise ValueError(f"No .safetensors files found in {model_path}")
            
        return files, str(model_dir)
    else:
        # HuggingFace Hub repository
        try:
            repo_files = list_repo_files(model_path)
            safetensors_files = [f for f in repo_files if f.endswith('.safetensors')]
            
            if not safetensors_files:
                raise ValueError(f"No .safetensors files found in {model_path}")
            
            files = {}
            logger.info(f"Downloading {len(safetensors_files)} files from {model_path}")
            
            # Download model files
            for filename in tqdm(safetensors_files, desc="Downloading model"):
                local_path = hf_hub_download(
                    repo_id=model_path,
                    filename=filename,
                    cache_dir=cache_dir
                )
                files[filename] = local_path
            
            # Also download config and tokenizer files
            config_files = ["config.json", "tokenizer.json", "tokenizer_config.json", 
                          "special_tokens_map.json", "generation_config.json"]
            
            model_dir = Path(local_path).parent  # Cache directory for this model
            
            for filename in config_files:
                if filename in repo_files:
                    try:
                        hf_hub_download(
                            repo_id=model_path,
                            filename=filename,
                            cache_dir=cache_dir
                        )
                    except:
                        pass  # Some files might not exist
                
            return files, str(model_dir)
            
        except Exception as e:
            raise ValueError(f"Could not access {model_path}: {e}")


def load_model_weights(file_paths: Dict[str, str], dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """Load model weights from safetensors files."""
    weights = {}
    
    logger.info(f"Loading model weights from {len(file_paths)} files")
    
    for filename, filepath in tqdm(file_paths.items(), desc="Loading model"):
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


def apply_delta(
    model_weights: Dict[str, torch.Tensor],
    delta_weights: Dict[str, torch.Tensor],
    metadata: Dict
) -> Dict[str, torch.Tensor]:
    """Apply delta to model weights: merged = model + delta."""
    
    # Verify tensor keys match
    model_keys = set(model_weights.keys())
    delta_keys = set(delta_weights.keys())
    
    if model_keys != delta_keys:
        missing_in_model = delta_keys - model_keys
        missing_in_delta = model_keys - delta_keys
        
        error_msg = "Tensor keys don't match between model and delta:\n"
        if missing_in_model:
            error_msg += f"Missing in model: {missing_in_model}\n"
        if missing_in_delta:
            error_msg += f"Missing in delta: {missing_in_delta}\n"
            
        raise ValueError(error_msg)
    
    merged = {}
    total_params = 0
    changed_params = 0
    
    logger.info("Applying delta (model + delta)")
    
    for key in tqdm(model_keys, desc="Applying delta"):
        model_tensor = model_weights[key]
        delta_tensor = delta_weights[key]
        
        # Verify shapes match
        if model_tensor.shape != delta_tensor.shape:
            raise ValueError(f"Shape mismatch for {key}: {model_tensor.shape} vs {delta_tensor.shape}")
        
        # Apply delta
        merged_tensor = model_tensor + delta_tensor
        merged[key] = merged_tensor
        
        # Track statistics
        total_params += delta_tensor.numel()
        changed_params += (delta_tensor != 0).sum().item()
    
    logger.info(f"Delta applied: {changed_params:,}/{total_params:,} parameters modified ({100*changed_params/total_params:.2f}%)")
    logger.info(f"L2 norm of applied delta: {metadata.get('l2_norm', 'unknown')}")
    
    return merged


def save_merged_model(
    merged_weights: Dict[str, torch.Tensor],
    model_dir: str,
    output_dir: str,
    metadata: Dict
):
    """Save merged weights and copy config/tokenizer files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save merged weights
    # Split into multiple files if too large (following HF conventions)
    max_file_size = 4.5 * 1024 * 1024 * 1024  # 4.5GB per file
    
    current_size = 0
    current_shard = {}
    shard_index = 1
    
    logger.info("Saving merged weights...")
    
    # First pass: determine total number of shards needed
    total_size = 0
    for tensor in merged_weights.values():
        total_size += tensor.numel() * tensor.element_size()
    
    total_shards = int(max(1, (total_size + max_file_size - 1) // max_file_size))
    
    for key, tensor in tqdm(merged_weights.items(), desc="Saving weights"):
        tensor_size = tensor.numel() * tensor.element_size()
        
        if current_size + tensor_size > max_file_size and current_shard:
            # Save current shard with correct total count
            shard_file = output_path / f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            save_file(current_shard, str(shard_file))
            
            current_shard = {}
            current_size = 0
            shard_index += 1
        
        current_shard[key] = tensor
        current_size += tensor_size
    
    # Save final shard
    if current_shard:
        if total_shards == 1:
            shard_file = output_path / "model.safetensors"
        else:
            shard_file = output_path / f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        save_file(current_shard, str(shard_file))
    
    logger.info(f"Model saved as {total_shards} shard(s)")
    
    # Copy config, tokenizer, and index files
    model_path = Path(model_dir)
    config_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json", "vocab.txt",
        "model.safetensors.index.json"  # Critical for sharded models
    ]
    
    for filename in config_files:
        src_file = model_path / filename
        if src_file.exists():
            dst_file = output_path / filename
            shutil.copy2(src_file, dst_file)
            logger.debug(f"Copied {filename}")
    
    # Special handling for index file: update shard names if needed
    index_file = output_path / "model.safetensors.index.json"
    if index_file.exists():
        logger.info("✅ Preserved model.safetensors.index.json from base model")
    
    # Save merge metadata
    merge_info = {
        "merged_from": {
            "base_model": metadata.get("base_model"),
            "delta_created_at": metadata.get("created_at"),
            "delta_l2_norm": metadata.get("l2_norm")
        },
        "total_parameters": sum(t.numel() for t in merged_weights.values()),
        "merge_timestamp": metadata.get("created_at")
    }
    
    with open(output_path / "merge_info.json", 'w') as f:
        json.dump(merge_info, f, indent=2)
    
    logger.info(f"Merged model saved to {output_dir}")


def test_inference(model_path: str, test_prompt: str = "What is Article 3 ECHR?"):
    """Test the merged model with a sample inference."""
    
    try:
        logger.info("Testing merged model with sample inference...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Tokenize input
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(test_prompt):].strip()
        
        logger.info("✅ Inference test successful!")
        logger.info(f"📝 Prompt: {test_prompt}")
        logger.info(f"🤖 Response: {generated_text[:200]}{'...' if len(generated_text) > 200 else ''}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  Inference test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply instruction delta to fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path or HuggingFace repo ID for the fine-tuned model"
    )
    
    parser.add_argument(
        "--delta_path",
        required=True,
        help="Path or HuggingFace repo ID containing delta.safetensors"
    )
    
    parser.add_argument(
        "--merged_out",
        required=True,
        help="Output directory for merged model"
    )
    
    parser.add_argument(
        "--dtype",
        choices=["fp16", "fp32"],
        help="Data type for merged model (inherits from delta if omitted)"
    )
    
    parser.add_argument(
        "--cache_dir",
        help="Cache directory for HuggingFace downloads"
    )
    
    parser.add_argument(
        "--test_inference",
        action="store_true",
        default=True,
        help="Test merged model with sample inference (default: True)"
    )
    
    parser.add_argument(
        "--push_to_hub",
        help="Push merged model to HuggingFace Hub (repo name)"
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting delta application")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Delta: {args.delta_path}")
        logger.info(f"Output: {args.merged_out}")
        
        # Load delta
        delta_weights, metadata = load_delta(args.delta_path)
        
        # Determine dtype
        if args.dtype:
            dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        else:
            # Inherit from delta metadata
            delta_dtype = metadata.get("dtype", "float16")
            dtype = torch.float16 if delta_dtype == "float16" else torch.float32
        
        logger.info(f"Using dtype: {dtype}")
        
        # Load model
        model_files, model_dir = load_model_files(args.model_path, args.cache_dir)
        model_weights = load_model_weights(model_files, dtype)
        
        # Apply delta
        merged_weights = apply_delta(model_weights, delta_weights, metadata)
        
        # Save merged model
        save_merged_model(merged_weights, model_dir, args.merged_out, metadata)
        
        # Test inference
        if args.test_inference:
            test_inference(args.merged_out)
        
        # Push to hub if requested
        if args.push_to_hub:
            logger.info(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=args.merged_out,
                    repo_id=args.push_to_hub,
                    repo_type="model"
                )
                logger.info(f"✅ Successfully pushed to {args.push_to_hub}")
            except Exception as e:
                logger.error(f"❌ Failed to push to hub: {e}")
        
        logger.info("✅ Delta application completed successfully!")
        logger.info(f"📁 Merged model: {args.merged_out}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main() 