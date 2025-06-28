"""
Production-scale attention collection for inference pipelines.
Clean, focused implementation with proper device handling.
"""

import json
import os
import sys
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import random

# Try to import torch and transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add project paths
inference_path = os.path.dirname(__file__)
vignettes_path = os.path.abspath(os.path.join(inference_path, '../vignettes'))
sys.path.extend([inference_path, vignettes_path])

from inference_pipeline import InferencePipeline
from utils import load_vignettes
from field_definitions import meta_prompt


class ProductionAttentionPipeline(InferencePipeline):
    """
    Production-scale inference pipeline with optional attention collection.
    Clean, focused implementation optimized for efficiency and correctness.
    """
    
    def __init__(self, model_subdir, models_base_dir="models", use_hf_hub=False, 
                 collect_attention=False, attention_sample_rate=0.1, 
                 storage_dir="attention_data"):
        """
        Initialize the production pipeline.
        
        Args:
            model_subdir (str): Model subdirectory or HF Hub model name
            models_base_dir (str): Base directory for local models
            use_hf_hub (bool): Whether to use HF Hub model
            collect_attention (bool): Whether to collect attention maps
            attention_sample_rate (float): Fraction of samples to collect attention for (0.0-1.0)
            storage_dir (str): Directory for attention data storage
        """
        # Initialize base pipeline first
        super().__init__(model_subdir, models_base_dir, use_hf_hub)
        
        # Attention collection settings
        self.collect_attention = collect_attention
        self.attention_sample_rate = attention_sample_rate
        self.storage_dir = storage_dir
        
        # Attention storage
        self.attention_count = 0
        self.attention_batch = []
        self.attention_batch_size = 50  # Smaller batches for memory efficiency
        self.h5_file = None
        self.metadata = {}
        
        # Initialize storage if needed
        if collect_attention:
            self._init_attention_storage()
            
        print(f"Production pipeline initialized:")
        print(f"  Attention collection: {'Enabled' if collect_attention else 'Disabled'}")
        if collect_attention:
            print(f"  Sample rate: {attention_sample_rate:.1%}")
            print(f"  Storage: {storage_dir}")
    
    def _init_attention_storage(self):
        """Initialize HDF5 storage for attention data."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_subdir)
        
        self.h5_path = os.path.join(self.storage_dir, f"attention_{model_name}_{timestamp}.h5")
        self.metadata_path = os.path.join(self.storage_dir, f"metadata_{model_name}_{timestamp}.json")
        
        self.metadata = {
            'model_name': model_name,
            'collection_start': datetime.now().isoformat(),
            'sample_rate': self.attention_sample_rate,
            'samples': {},
            'total_samples_collected': 0
        }
        
        print(f"  Attention storage: {self.h5_path}")
    
    def _should_collect_attention(self, sample_idx: int, total_samples: int) -> bool:
        """Determine whether to collect attention for this sample."""
        if not self.collect_attention or self.attention_sample_rate <= 0:
            return False
        
        if self.attention_sample_rate >= 1.0:
            return True
        
        # Simple deterministic sampling
        target_samples = int(total_samples * self.attention_sample_rate)
        if target_samples == 0:
            return False
        
        interval = max(1, total_samples // target_samples)
        return sample_idx % interval == 0
    
    def _run_inference_with_attention(self, prompt: str, sample_metadata: Dict) -> Tuple[str, Optional[Dict]]:
        """Run inference and capture attention weights."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )
            
            # Handle device placement properly
            if hasattr(self, 'using_device_map') and self.using_device_map:
                # When using device_map, move ALL input components to the device where embeddings are located
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Standard single-device placement
                inputs = inputs.to(self.device)
            
            # Generate with attention collection
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,
                    return_dict_in_generate=True
                )
            
            # Extract response text
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs.sequences[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Extract attention if available
            attention_data = None
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_data = self._extract_attention(outputs.attentions, inputs, sample_metadata)
            
            return response_text.strip(), attention_data
            
        except Exception as e:
            print(f"Error during inference with attention: {str(e)}")
            return f"ERROR: {str(e)}", None
    
    def _extract_attention(self, attentions: Tuple, inputs: Dict, metadata: Dict) -> Optional[Dict]:
        """Extract essential attention data efficiently."""
        try:
            input_ids = inputs['input_ids'][0].cpu().numpy()
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Store minimal essential data
            attention_data = {
                'sample_id': metadata.get('sample_id'),
                'input_length': len(input_tokens),
                'input_tokens': input_tokens,
                'attention_matrices': []
            }
            
            # Extract attention from first few generation steps only
            max_steps = min(2, len(attentions))  # Only first 2 steps to save space
            key_layers = [0, -1]  # First and last layers only
            
            for step_idx in range(max_steps):
                step_attention = attentions[step_idx]
                
                for layer_idx in key_layers:
                    if abs(layer_idx) < len(step_attention):
                        layer_attn = step_attention[layer_idx]  # [batch, heads, seq_len, seq_len]
                        
                        # Extract attention from generated token to input tokens
                        if layer_attn.shape[2] > len(input_tokens):
                            gen_pos = len(input_tokens) + step_idx
                            if gen_pos < layer_attn.shape[2]:
                                attn_weights = layer_attn[0, :, gen_pos, :len(input_tokens)].cpu().numpy()
                                
                                attention_data['attention_matrices'].append({
                                    'step': step_idx,
                                    'layer': layer_idx,
                                    'weights': attn_weights.astype(np.float16)  # Compress
                                })
            
            return attention_data if attention_data['attention_matrices'] else None
            
        except Exception as e:
            print(f"Error extracting attention: {str(e)}")
            return None
    
    def _add_attention_sample(self, attention_data: Dict, sample_metadata: Dict):
        """Add attention sample to batch for efficient storage."""
        if attention_data:
            self.attention_batch.append(attention_data)
            
            # Store metadata
            self.metadata['samples'][attention_data['sample_id']] = {
                'topic': sample_metadata.get('topic'),
                'meta_topic': sample_metadata.get('meta_topic'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save batch when full
            if len(self.attention_batch) >= self.attention_batch_size:
                self._save_attention_batch()
    
    def _save_attention_batch(self):
        """Save accumulated attention data to HDF5."""
        if not self.attention_batch:
            return
        
        # Initialize HDF5 file on first write
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'w')
        
        # Save each sample
        for attention_data in self.attention_batch:
            sample_id = attention_data['sample_id']
            sample_group = self.h5_file.create_group(f'sample_{sample_id}')
            
            # Store metadata
            sample_group.attrs['input_length'] = attention_data['input_length']
            
            # Store input tokens
            dt = h5py.special_dtype(vlen=str)
            sample_group.create_dataset('input_tokens', data=attention_data['input_tokens'], dtype=dt)
            
            # Store attention matrices
            for i, attn_matrix in enumerate(attention_data['attention_matrices']):
                dataset_name = f"attention_{i}"
                sample_group.create_dataset(
                    dataset_name,
                    data=attn_matrix['weights'],
                    compression='gzip',
                    compression_opts=9,
                    dtype=np.float16
                )
                sample_group[dataset_name].attrs['step'] = attn_matrix['step']
                sample_group[dataset_name].attrs['layer'] = attn_matrix['layer']
        
        # Flush to disk
        self.h5_file.flush()
        
        self.attention_count += len(self.attention_batch)
        print(f"Saved attention batch: {len(self.attention_batch)} samples (total: {self.attention_count})")
        self.attention_batch = []
    
    def _run_inference_batch_optimized(self, prompts: List[str], batch_size: int = 16) -> List[str]:
        """
        Optimized batch inference with proper device handling.
        Works correctly with both single-GPU and device_map setups.
        """
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                )
                
                # Handle device placement properly
                if hasattr(self, 'using_device_map') and self.using_device_map:
                    # When using device_map, move ALL input components to the device where embeddings are located
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    # Standard single-device placement
                    inputs = inputs.to(self.device)
                
                # Generate responses
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.7,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        output_attentions=False,  # No attention for batch processing
                        output_hidden_states=False,
                        output_scores=False
                    )
                
                # Decode responses
                batch_responses = []
                for j, output_ids in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    response_tokens = output_ids[input_length:]
                    response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    batch_responses.append(response.strip())
                
                all_responses.extend(batch_responses)
                
                # Progress indicator
                if len(prompts) > batch_size * 2:
                    processed = min(i + batch_size, len(prompts))
                    print(f"    Batch progress: {processed}/{len(prompts)} prompts processed")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error during batch inference: {str(e)}")
                # Fallback to individual processing
                for prompt in batch_prompts:
                    try:
                        response = super()._run_inference(prompt)
                        all_responses.append(response)
                    except Exception as individual_error:
                        print(f"Error in individual fallback: {str(individual_error)}")
                        all_responses.append(f"ERROR: {str(individual_error)}")
        
        return all_responses
    
    def generate_inference_records(self, vignettes: List[Dict], batch_size: int = 16) -> List[Dict]:
        """
        Main method to generate inference records with optional attention collection.
        Clean, efficient implementation that works reliably.
        
        Args:
            vignettes: List of vignette dictionaries
            batch_size: Number of samples to process in each batch
        """
        print(f"🚀 Starting production inference...")
        print(f"   Batch size: {batch_size}")
        print(f"   Attention collection: {'Enabled' if self.collect_attention else 'Disabled'}")
        
        # Phase 1: Generate all prompts and metadata
        print("📋 Phase 1: Generating prompts...")
        all_prompts = []
        all_sample_data = []
        
        total_permutations = self._count_total_permutations(vignettes)
        current_permutation = 0
        
        print(f"   Total permutations: {total_permutations:,}")
        
        # Generate all prompts using existing logic from base class
        for vignette_idx, vignette in enumerate(vignettes):
            print(f"  Processing vignette {vignette_idx + 1}/{len(vignettes)}: {vignette['topic']}")
            
            # Import required modules
            from field_definitions import (get_name_for_country_gender, get_pronoun, systems_to_countries_map, 
                                          safety_to_countries_map, safety_and_systems_to_countries_map, get_verb_present_third_person, 
                                          get_verb_past_be, get_pronoun_possessive)
            from utils import resolve_field_reference
            from itertools import product

            generic_fields = vignette.get('generic_fields', {})
            ordinal_fields = vignette.get('ordinal_fields', {})
            horizontal_fields = vignette.get('horizontal_fields', {})
            derived_fields = vignette.get('derived_fields', {})

            generic_keys = list(generic_fields.keys())
            generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_keys]

            ordinal_keys = list(ordinal_fields.keys())
            ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_keys]

            horizontal_keys = list(horizontal_fields.keys())
            horizontal_lists = [horizontal_fields[k] for k in horizontal_keys]

            for generic_vals in product(*generic_lists):
                for ordinal_vals in product(*ordinal_lists) if ordinal_lists else [()]:
                    for horizontal_vals in product(*horizontal_lists) if horizontal_lists else [()]:
                        current_permutation += 1
                        
                        # Progress indicator
                        if current_permutation % 5000 == 0:
                            progress = current_permutation/total_permutations*100
                            print(f"    Progress: {current_permutation:,}/{total_permutations:,} ({progress:.1f}%)")
                        
                        # Build sample values
                        sample_values = {}
                        
                        for k, v in zip(generic_keys, generic_vals):
                            sample_values[k] = v
                        for k, v in zip(ordinal_keys, ordinal_vals):
                            sample_values[k] = v
                            sample_values[f"{k}__ordinal"] = ordinal_fields[k][v]
                        for k, v in zip(horizontal_keys, horizontal_vals):
                            sample_values[k] = v
                        
                        # Handle derived fields
                        if derived_fields:
                            for dfield, dspec in derived_fields.items():
                                if dfield == "name" and "country" in sample_values and "gender" in sample_values:
                                    sample_values["name"] = get_name_for_country_gender(sample_values["country"], sample_values["gender"])
                                elif dfield == "country_B":
                                    mapping = dspec["mapping"]
                                    source_field = dspec["source_field"]
                                    if mapping == "systems_to_countries_map":
                                        val = sample_values.get(source_field)
                                        if val and val in systems_to_countries_map:
                                            sample_values["country_B"] = systems_to_countries_map[val][0]
                                    elif mapping == "safety_to_countries_map":
                                        val = sample_values.get(source_field)
                                        if val and val in safety_to_countries_map:
                                            sample_values["country_B"] = safety_to_countries_map[val][0]
                                    elif mapping == "safety_and_systems_to_countries_map":
                                        val = sample_values.get(source_field)
                                        if val and val in safety_and_systems_to_countries_map:
                                            sample_values["country_B"] = safety_and_systems_to_countries_map[val][0]
                        
                        # Add pronoun and grammar helpers
                        if 'gender' in sample_values:
                            gender = sample_values['gender']
                            sample_values['pronoun'] = get_pronoun(gender)
                            sample_values['pronoun_was_were'] = get_verb_past_be(gender)
                            sample_values['pronoun_possessive'] = get_pronoun_possessive(gender)
                            sample_values['pronoun_suffers'] = get_verb_present_third_person(gender, 'suffer')
                            sample_values['pronoun_lives'] = get_verb_present_third_person(gender, 'live')
                            sample_values['pronoun_works'] = get_verb_present_third_person(gender, 'work')
                        
                        # Generate vignette text
                        try:
                            vignette_text = vignette["vignette_template"].format(**sample_values)
                        except KeyError as e:
                            print(f"Warning: Missing field {e} for vignette template")
                            continue
                        
                        # Create prompt
                        prompt = self._create_prompt(vignette_text)
                        all_prompts.append(prompt)
                        
                        # Store sample data for final record construction
                        sample_data = {
                            'sample_id': current_permutation,
                            'meta_topic': vignette['meta_topic'],
                            'topic': vignette['topic'],
                            'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                            'vignette_text': vignette_text,
                            'should_collect_attention': self._should_collect_attention(current_permutation, total_permutations)
                        }
                        
                        # Add ordinal values
                        for k in ordinal_keys:
                            if f"{k}__ordinal" in sample_values:
                                sample_data[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
                        
                        all_sample_data.append(sample_data)
        
        print(f"✅ Generated {len(all_prompts):,} prompts")
        
        # Phase 2: Run inference
        print(f"🔄 Phase 2: Running inference...")
        
        # Use batch processing for efficiency, but collect attention individually when needed
        all_responses = []
        
        if self.collect_attention:
            # When collecting attention, we need to process individually for samples that need attention
            print("  Running with attention collection...")
            
            for i, (prompt, sample_data) in enumerate(zip(all_prompts, all_sample_data)):
                if sample_data['should_collect_attention']:
                    # Individual processing with attention
                    response, attention_data = self._run_inference_with_attention(prompt, sample_data)
                    if attention_data:
                        self._add_attention_sample(attention_data, sample_data)
                else:
                    # Fast processing without attention
                    response = super()._run_inference(prompt)
                
                all_responses.append(response)
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"    Progress: {i + 1:,}/{len(all_prompts):,} ({(i + 1)/len(all_prompts)*100:.1f}%)")
        else:
            # Fast batch processing without attention
            print("  Running optimized batch inference...")
            all_responses = self._run_inference_batch_optimized(all_prompts, batch_size)
        
        # Phase 3: Build final records
        print("📦 Phase 3: Building final records...")
        records = []
        
        for i, (sample_data, response) in enumerate(zip(all_sample_data, all_responses)):
            record = {
                **sample_data,
                'model_response': response,
                'inference_timestamp': datetime.now().isoformat()
            }
            # Remove internal fields
            record.pop('should_collect_attention', None)
            records.append(record)
        
        # Finalize attention collection
        if self.collect_attention:
            self._finalize_attention_collection()
        
        print(f"✅ Inference completed!")
        print(f"   Total records: {len(records):,}")
        if self.collect_attention:
            print(f"   Attention samples: {self.attention_count:,}")
        
        return records
    
    def _finalize_attention_collection(self):
        """Finalize attention data collection and close files."""
        if not self.collect_attention:
            return
            
        # Save any remaining batch
        if self.attention_batch:
            self._save_attention_batch()
        
        # Close HDF5 file
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
        
        # Save metadata
        self.metadata['collection_end'] = datetime.now().isoformat()
        self.metadata['total_samples_collected'] = self.attention_count
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        if self.attention_count > 0:
            file_size_mb = os.path.getsize(self.h5_path) / (1024 * 1024)
            print(f"✅ Attention collection finalized:")
            print(f"   Samples collected: {self.attention_count}")
            print(f"   Storage size: {file_size_mb:.1f} MB")
            print(f"   Data file: {self.h5_path}")
            print(f"   Metadata: {self.metadata_path}")
    
    def _count_total_permutations(self, vignettes: List[Dict]) -> int:
        """Count total permutations efficiently."""
        from utils import resolve_field_reference
        
        total_count = 0
        
        for vignette in vignettes:
            generic_fields = vignette.get('generic_fields', {})
            ordinal_fields = vignette.get('ordinal_fields', {})
            horizontal_fields = vignette.get('horizontal_fields', {})
            
            generic_count = 1
            for field_spec in generic_fields.values():
                field_list = resolve_field_reference(field_spec)
                generic_count *= len(field_list)
            
            ordinal_count = 1
            for ordinal_values in ordinal_fields.values():
                ordinal_count *= len(ordinal_values)
            
            horizontal_count = 1
            for horizontal_list in horizontal_fields.values():
                horizontal_count *= len(horizontal_list)
            
            vignette_total = generic_count * ordinal_count * horizontal_count
            total_count += vignette_total
        
        return total_count


def main():
    """Command line interface for production inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production inference with optional attention collection")
    parser.add_argument("model_subdir", help="Model subdirectory or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", required=True, help="Output path for inference results")
    parser.add_argument("--attention-dir", default="attention_data",
                       help="Directory to save attention data")
    parser.add_argument("--attention-rate", type=float, default=0.1,
                       help="Fraction of samples to collect attention for (0.0-1.0)")
    parser.add_argument("--collect-attention", action="store_true",
                       help="Enable attention collection")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Load model from Hugging Face Hub")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProductionAttentionPipeline(
        args.model_subdir,
        use_hf_hub=args.use_hf_hub,
        collect_attention=args.collect_attention,
        attention_sample_rate=args.attention_rate,
        storage_dir=args.attention_dir
    )
    
    # Load vignettes
    vignettes = load_vignettes(args.vignettes)
    
    # Run inference
    records = pipeline.generate_inference_records(vignettes, batch_size=args.batch_size)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Production run completed!")
    print(f"Results saved to: {args.output}")
    if args.collect_attention:
        print(f"Attention data: {args.attention_dir}")


if __name__ == "__main__":
    main() 