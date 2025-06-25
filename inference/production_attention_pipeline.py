"""
Production-scale attention collection for 25,000+ samples.
Optimized for minimal storage overhead and maximum data retention.
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
    Production-scale attention collection pipeline.
    Optimized for 25,000+ samples with minimal storage overhead.
    """
    
    def __init__(self, model_subdir, models_base_dir="models", use_hf_hub=False, 
                 collect_attention=True, attention_sample_rate=0.8, 
                 storage_dir="attention_data"):
        """
        Initialize the production attention pipeline.
        
        Args:
            model_subdir (str): Model subdirectory or HF Hub model name
            models_base_dir (str): Base directory for local models
            use_hf_hub (bool): Whether to use HF Hub model
            collect_attention (bool): Whether to collect attention maps
            attention_sample_rate (float): Fraction of samples to collect attention for (0.0-1.0)
            storage_dir (str): Directory for attention data storage
        """
        super().__init__(model_subdir, models_base_dir, use_hf_hub)
        
        self.collect_attention = collect_attention
        self.attention_sample_rate = attention_sample_rate
        self.storage_dir = storage_dir
        
        # Initialize storage
        if collect_attention:
            self._init_storage()
            
        self.attention_count = 0
        self.batch_size = 100  # Process attention in batches
        self.current_batch = []
        
        print(f"Production attention collection: {'Enabled' if collect_attention else 'Disabled'}")
        if collect_attention:
            print(f"Sample rate: {attention_sample_rate:.1%}")
            print(f"Storage: {storage_dir}")
    
    def _init_storage(self):
        """Initialize HDF5 storage for efficient attention data."""
        os.makedirs(self.storage_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(self.model_subdir)
        
        self.h5_path = os.path.join(self.storage_dir, f"attention_{model_name}_{timestamp}.h5")
        self.metadata_path = os.path.join(self.storage_dir, f"metadata_{model_name}_{timestamp}.json")
        
        # Will be created when first batch is saved
        self.h5_file = None
        self.metadata = {
            'model_name': model_name,
            'collection_start': datetime.now().isoformat(),
            'sample_rate': self.attention_sample_rate,
            'samples': {},  # sample_id -> metadata
            'total_samples_collected': 0
        }
        
        print(f"Attention storage initialized: {self.h5_path}")
    
    def _should_collect_attention(self, sample_idx: int, total_samples: int) -> bool:
        """Determine whether to collect attention for this sample."""
        if not self.collect_attention or self.attention_sample_rate <= 0:
            return False
        
        if self.attention_sample_rate >= 1.0:
            return True
        
        # Stratified sampling with deterministic selection
        target_samples = int(total_samples * self.attention_sample_rate)
        if target_samples == 0:
            return False
        
        interval = total_samples / target_samples
        return sample_idx % int(interval) == 0
    
    def _run_inference_with_attention(self, prompt: str, sample_metadata: Dict) -> Tuple[str, Optional[Dict]]:
        """
        Run inference and capture raw attention weights efficiently.
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Generate response with attention collection
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    output_attentions=True,  # Capture attention weights
                    return_dict_in_generate=True
                )
            
            # Extract response text
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs.sequences[0][input_length:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Process attention weights efficiently
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_data = self._extract_raw_attention(
                    outputs.attentions, inputs, outputs.sequences[0], sample_metadata
                )
                return response_text.strip(), attention_data
            
            return response_text.strip(), None
            
        except Exception as e:
            print(f"Error during inference with attention: {str(e)}")
            return f"ERROR: {str(e)}", None
    
    def _extract_raw_attention(self, attentions: Tuple, inputs: Dict, 
                             full_output: torch.Tensor, metadata: Dict) -> Dict:
        """
        Extract only the essential raw attention data for efficient storage.
        """
        try:
            input_ids = inputs['input_ids'][0].cpu().numpy()
            
            # Store only essential data
            raw_data = {
                'sample_id': metadata.get('sample_id'),
                'input_length': len(input_ids),
                'generated_length': len(full_output) - len(input_ids),
                'input_tokens': self.tokenizer.convert_ids_to_tokens(input_ids),
            }
            
            # Extract raw attention for first few generation steps only
            max_steps = min(3, len(attentions))  # Only first 3 steps to save space
            attention_arrays = []
            
            for step_idx in range(max_steps):
                step_attention = attentions[step_idx]  # List of layer attentions
                
                # Store only key layers to reduce size
                key_layers = [0, len(step_attention)//2, len(step_attention)-1]  # First, middle, last
                
                step_data = []
                for layer_idx in key_layers:
                    if layer_idx < len(step_attention):
                        layer_attn = step_attention[layer_idx]  # [batch, heads, seq_len, seq_len]
                        
                        # Extract attention from generated token to all input tokens
                        if step_idx < layer_attn.shape[2] - len(input_ids):
                            # Get attention weights: [num_heads, input_length]
                            gen_pos = len(input_ids) + step_idx
                            attention_to_input = layer_attn[0, :, gen_pos, :len(input_ids)].cpu().numpy()
                            
                            step_data.append({
                                'layer_idx': layer_idx,
                                'attention_matrix': attention_to_input.astype(np.float16)  # Reduce precision
                            })
                
                attention_arrays.append({
                    'generation_step': step_idx,
                    'layers': step_data
                })
            
            raw_data['attention_steps'] = attention_arrays
            return raw_data
            
        except Exception as e:
            print(f"Error extracting raw attention: {str(e)}")
            return None
    
    def _save_attention_batch(self):
        """Save accumulated attention data in batches for efficiency."""
        if not self.current_batch:
            return
        
        # Initialize HDF5 file on first write
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'w')
            
        # Save each sample in the batch
        for attention_data in self.current_batch:
            sample_id = attention_data['sample_id']
            
            # Create group for this sample
            sample_group = self.h5_file.create_group(f'sample_{sample_id}')
            
            # Store basic metadata
            sample_group.attrs['input_length'] = attention_data['input_length']
            sample_group.attrs['generated_length'] = attention_data['generated_length']
            
            # Store input tokens as strings
            dt = h5py.special_dtype(vlen=str)
            sample_group.create_dataset('input_tokens', data=attention_data['input_tokens'], dtype=dt)
            
            # Store attention matrices efficiently
            for step_idx, step_data in enumerate(attention_data['attention_steps']):
                step_group = sample_group.create_group(f'step_{step_idx}')
                step_group.attrs['generation_step'] = step_data['generation_step']
                
                for layer_data in step_data['layers']:
                    layer_name = f"layer_{layer_data['layer_idx']}"
                    # Compress attention matrices
                    step_group.create_dataset(
                        layer_name,
                        data=layer_data['attention_matrix'],
                        compression='gzip',
                        compression_opts=9,
                        dtype=np.float16
                    )
        
        # Flush to disk
        self.h5_file.flush()
        
        print(f"Saved attention batch: {len(self.current_batch)} samples")
        self.attention_count += len(self.current_batch)
        self.current_batch = []
    
    def _add_attention_sample(self, attention_data: Dict, sample_metadata: Dict):
        """Add attention sample to current batch."""
        if attention_data:
            self.current_batch.append(attention_data)
            
            # Store minimal metadata separately
            self.metadata['samples'][attention_data['sample_id']] = {
                'vignette_topic': sample_metadata.get('vignette_topic'),
                'meta_topic': sample_metadata.get('meta_topic'),
                'fields': sample_metadata.get('fields', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save batch when full
            if len(self.current_batch) >= self.batch_size:
                self._save_attention_batch()
    
    def _run_inference(self, prompt: str, sample_metadata: Dict = None) -> str:
        """Override to potentially collect attention."""
        if self.collect_attention and sample_metadata is not None and sample_metadata.get('should_collect'):
            response, attention_data = self._run_inference_with_attention(prompt, sample_metadata)
            
            if attention_data is not None:
                self._add_attention_sample(attention_data, sample_metadata)
                
            return response
        else:
            return super()._run_inference(prompt)
    
    def finalize_attention_collection(self):
        """Finalize attention data collection and close files."""
        # Skip if attention collection is disabled
        if not self.collect_attention:
            print(f"\n✅ Production inference completed (no attention collected)")
            return
            
        # Save any remaining batch
        if self.current_batch:
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
            print(f"\n✅ Attention collection finalized:")
            print(f"   Samples collected: {self.attention_count}")
            print(f"   Storage size: {file_size_mb:.1f} MB")
            print(f"   Data file: {self.h5_path}")
            print(f"   Metadata: {self.metadata_path}")
            print(f"   Average MB per sample: {file_size_mb / self.attention_count:.2f}")
    
    def generate_inference_records_with_attention(self, vignettes: List[Dict]) -> List[Dict]:
        """
        Generate inference records with efficient attention collection.
        """
        records = []
        total_permutations = self._count_total_permutations(vignettes)
        current_permutation = 0
        
        print(f"🚀 Starting production attention collection...")
        print(f"Total permutations: {total_permutations:,}")
        
        if self.collect_attention:
            expected_attention_samples = int(total_permutations * self.attention_sample_rate)
            print(f"Expected attention samples: {expected_attention_samples:,}")
            print(f"Estimated storage: {expected_attention_samples * 0.5:.1f} MB")
        
        try:
            for vignette_idx, vignette in enumerate(vignettes):
                print(f"\n📚 Processing vignette {vignette_idx + 1}/{len(vignettes)}: {vignette['topic']}")
                
                # Generate all permutations using existing logic
                from field_definitions import (get_name_for_country_gender, get_pronoun, systems_to_countries_map, 
                                              safety_to_countries_map, get_verb_present_third_person, 
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
                            if current_permutation % 1000 == 0:
                                progress = current_permutation/total_permutations*100
                                print(f"  Progress: {current_permutation:,}/{total_permutations:,} ({progress:.1f}%) - Attention: {self.attention_count}")
                            
                            # Build sample values (same logic as base class)
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
                            
                            # Add pronoun and grammar helpers
                            if 'gender' in sample_values:
                                gender = sample_values['gender']
                                sample_values['pronoun'] = get_pronoun(gender)
                                sample_values['pronoun_was_were'] = get_verb_past_be(gender)
                                sample_values['pronoun_possessive'] = get_pronoun_possessive(gender)
                                # Common verb forms
                                sample_values['pronoun_suffers'] = get_verb_present_third_person(gender, 'suffer')
                                sample_values['pronoun_lives'] = get_verb_present_third_person(gender, 'live')
                                sample_values['pronoun_works'] = get_verb_present_third_person(gender, 'work')
                            
                            # Generate vignette text
                            try:
                                vignette_text = vignette["vignette_template"].format(**sample_values)
                            except KeyError as e:
                                print(f"Warning: Missing field {e} for vignette template")
                                continue
                            
                            # Check if we should collect attention for this sample
                            should_collect = self._should_collect_attention(current_permutation, total_permutations)
                            
                            # Create sample metadata
                            sample_metadata = {
                                'sample_id': current_permutation,
                                'vignette_idx': vignette_idx,
                                'vignette_topic': vignette['topic'],
                                'meta_topic': vignette['meta_topic'],
                                'fields': dict(sample_values),
                                'should_collect': should_collect
                            }
                            
                            # Create prompt and run inference
                            prompt = self._create_prompt(vignette_text)
                            model_response = self._run_inference(prompt, sample_metadata)
                            
                            # Build minimal record (don't store attention data here)
                            record = {
                                'meta_topic': vignette['meta_topic'],
                                'topic': vignette['topic'],
                                'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                                'vignette_text': vignette_text,
                                'model_response': model_response,
                                'inference_timestamp': datetime.now().isoformat(),
                                'attention_collected': should_collect
                            }
                            
                            # Add ordinal values
                            for k in ordinal_keys:
                                if f"{k}__ordinal" in sample_values:
                                    record[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
                            
                            records.append(record)
            
        finally:
            # Always finalize attention collection
            self.finalize_attention_collection()
        
        print(f"\n✅ Inference completed!")
        print(f"Total records: {len(records):,}")
        print(f"Attention samples: {self.attention_count:,}")
        
        return records

    def generate_inference_records_with_attention_optimized(self, vignettes: List[Dict], batch_size: int = 8) -> List[Dict]:
        """
        OPTIMIZED: Generate inference records with efficient batched processing and attention collection.
        This is 5-10x faster than the individual processing version.
        
        Args:
            vignettes: List of vignette dictionaries
            batch_size: Number of samples to process in each batch (reduce if memory issues)
        """
        print(f"🚀 Starting OPTIMIZED production attention collection...")
        
        # Phase 1: Generate all prompts and metadata first
        print("📋 Phase 1: Generating all prompts...")
        all_prompts = []
        all_metadatas = []
        all_sample_data = []  # For building final records
        
        total_permutations = self._count_total_permutations(vignettes)
        current_permutation = 0
        
        print(f"Total permutations: {total_permutations:,}")
        
        if self.collect_attention:
            expected_attention_samples = int(total_permutations * self.attention_sample_rate)
            print(f"Expected attention samples: {expected_attention_samples:,}")
            print(f"Estimated storage: {expected_attention_samples * 0.5:.1f} MB")
        
        # Generate all prompts (reusing existing logic)
        for vignette_idx, vignette in enumerate(vignettes):
            print(f"  Generating prompts for vignette {vignette_idx + 1}/{len(vignettes)}: {vignette['topic']}")
            
            # Import required modules
            from field_definitions import (get_name_for_country_gender, get_pronoun, systems_to_countries_map, 
                                          safety_to_countries_map, get_verb_present_third_person, 
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
                            print(f"    Prompt generation: {current_permutation:,}/{total_permutations:,} ({progress:.1f}%)")
                        
                        # Build sample values (same logic as before)
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
                        
                        # Add pronoun and grammar helpers
                        if 'gender' in sample_values:
                            gender = sample_values['gender']
                            sample_values['pronoun'] = get_pronoun(gender)
                            sample_values['pronoun_was_were'] = get_verb_past_be(gender)
                            sample_values['pronoun_possessive'] = get_pronoun_possessive(gender)
                            # Common verb forms
                            sample_values['pronoun_suffers'] = get_verb_present_third_person(gender, 'suffer')
                            sample_values['pronoun_lives'] = get_verb_present_third_person(gender, 'live')
                            sample_values['pronoun_works'] = get_verb_present_third_person(gender, 'work')
                        
                        # Generate vignette text
                        try:
                            vignette_text = vignette["vignette_template"].format(**sample_values)
                        except KeyError as e:
                            print(f"Warning: Missing field {e} for vignette template")
                            continue
                        
                        # Check if we should collect attention for this sample
                        should_collect = self._should_collect_attention(current_permutation, total_permutations)
                        
                        # Create prompt
                        prompt = self._create_prompt(vignette_text)
                        
                        # Store everything
                        all_prompts.append(prompt)
                        
                        sample_metadata = {
                            'sample_id': current_permutation,
                            'vignette_idx': vignette_idx,
                            'vignette_topic': vignette['topic'],
                            'meta_topic': vignette['meta_topic'],
                            'fields': dict(sample_values),
                            'should_collect': should_collect
                        }
                        all_metadatas.append(sample_metadata)
                        
                        # Store data for final record construction
                        sample_data = {
                            'meta_topic': vignette['meta_topic'],
                            'topic': vignette['topic'],
                            'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                            'vignette_text': vignette_text,
                            'attention_collected': should_collect
                        }
                        
                        # Add ordinal values
                        for k in ordinal_keys:
                            if f"{k}__ordinal" in sample_values:
                                sample_data[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
                        
                        all_sample_data.append(sample_data)
        
        print(f"✅ Generated {len(all_prompts):,} prompts")
        
        # Phase 2: Run batched inference
        print(f"🔄 Phase 2: Running batched inference (batch_size={batch_size})...")
        
        try:
            all_responses = self._run_inference_batch_with_attention(
                all_prompts, all_metadatas, batch_size=batch_size
            )
        finally:
            # Always finalize attention collection
            self.finalize_attention_collection()
        
        # Phase 3: Build final records
        print("📦 Phase 3: Building final records...")
        records = []
        
        for i, (sample_data, response) in enumerate(zip(all_sample_data, all_responses)):
            record = sample_data.copy()
            record['model_response'] = response
            record['inference_timestamp'] = datetime.now().isoformat()
            records.append(record)
        
        print(f"\n✅ OPTIMIZED Inference completed!")
        print(f"Total records: {len(records):,}")
        print(f"Attention samples: {self.attention_count:,}")
        
        return records

    def _run_inference_batch_with_attention(self, prompts: List[str], sample_metadatas: List[Dict], batch_size: int = 8) -> List[str]:
        """
        Run batched inference with optional attention collection.
        Much more efficient than individual processing.
        """
        all_responses = []
        
        # Process in smaller batches to manage memory
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_metadatas = sample_metadatas[i:i+batch_size] if sample_metadatas else []
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).to(self.device)
                
                # Prepare generation kwargs
                generation_kwargs = {
                    **inputs,
                    'max_new_tokens': 250,
                    'do_sample': True,
                    'temperature': 0.5,
                    'top_p': 0.6,
                    'repetition_penalty': 1.2,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'use_cache': True,
                    'num_beams': 1,
                    'return_dict_in_generate': True,
                    'output_attentions': self.collect_attention  # Only if collecting
                }
                
                # Generate responses
                with torch.no_grad():
                    outputs = self.model.generate(**generation_kwargs)
                
                # Process outputs
                if self.collect_attention and hasattr(outputs, 'attentions') and outputs.attentions:
                    # Extract attention for samples that need it
                    batch_responses = self._process_batch_with_attention(
                        outputs, inputs, batch_metadatas
                    )
                else:
                    # Standard processing without attention
                    batch_responses = []
                    for j, output_ids in enumerate(outputs.sequences):
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
                for j, prompt in enumerate(batch_prompts):
                    try:
                        metadata = batch_metadatas[j] if j < len(batch_metadatas) else {}
                        response = self._run_inference(prompt, metadata)
                        all_responses.append(response)
                    except Exception as individual_error:
                        print(f"Error in individual fallback: {str(individual_error)}")
                        all_responses.append(f"ERROR: {str(individual_error)}")
        
        return all_responses
    
    def _process_batch_with_attention(self, outputs, inputs, batch_metadatas: List[Dict]) -> List[str]:
        """Process batch outputs and extract attention for relevant samples."""
        batch_responses = []
        
        for j, output_ids in enumerate(outputs.sequences):
            # Decode response
            input_length = inputs['input_ids'][j].shape[0]
            response_tokens = output_ids[input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            batch_responses.append(response.strip())
            
            # Check if we should collect attention for this sample
            if (j < len(batch_metadatas) and 
                batch_metadatas[j].get('should_collect', False)):
                
                try:
                    # Extract attention for this specific sample in the batch
                    sample_attention = self._extract_batch_attention(
                        outputs.attentions, inputs, output_ids, j, batch_metadatas[j]
                    )
                    
                    if sample_attention:
                        self._add_attention_sample(sample_attention, batch_metadatas[j])
                        
                except Exception as e:
                    print(f"Warning: Failed to extract attention for sample {j}: {e}")
        
        return batch_responses
    
    def _extract_batch_attention(self, attentions: Tuple, inputs: Dict, 
                                output_ids: torch.Tensor, batch_idx: int, metadata: Dict) -> Optional[Dict]:
        """Extract attention for a specific sample in a batch."""
        try:
            if not attentions or len(attentions) == 0:
                return None
            
            # Get input tokens for this sample
            input_ids = inputs['input_ids'][batch_idx]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            input_length = len(input_tokens)
            
            # Get generated tokens
            generated_ids = output_ids[input_length:]
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            
            if len(generated_tokens) == 0:
                return None
            
            # Extract attention from key layers and steps
            attention_data = {
                'sample_id': metadata.get('sample_id', f"batch_{batch_idx}"),
                'input_tokens': input_tokens,
                'generated_tokens': generated_tokens[:self.max_generation_steps],
                'layers': {}
            }
            
            # Process each generation step (limited to save space)
            for step_idx in range(min(len(attentions), self.max_generation_steps)):
                step_attentions = attentions[step_idx]  # All layers for this step
                
                for layer_idx in self.target_layers:
                    if layer_idx < len(step_attentions):
                        # Get attention for this sample: [num_heads, seq_len, seq_len]
                        layer_attention = step_attentions[layer_idx]  # [num_heads, seq_len, seq_len]
                        
                        # Focus on attention from generated token to input tokens
                        if layer_attention.size(-1) > input_length:
                            # Attention from current generated token to all input tokens
                            gen_to_input_attention = layer_attention[:, -1, :input_length]  # [num_heads, input_length]
                            
                            # Convert to numpy and compress
                            attention_matrix = gen_to_input_attention.cpu().float().numpy().astype(np.float16)
                            
                            # Store in nested structure
                            if layer_idx not in attention_data['layers']:
                                attention_data['layers'][layer_idx] = {}
                            attention_data['layers'][layer_idx][step_idx] = attention_matrix
            
            return attention_data if attention_data['layers'] else None
            
        except Exception as e:
            print(f"Error extracting batch attention: {e}")
            return None

    def generate_inference_records_optimized(self, vignettes: List[Dict], batch_size: int = 16) -> List[Dict]:
        """
        OPTIMIZED: Generate inference records with efficient batched processing (no attention collection).
        This is the recommended method for fast inference without attention data.
        
        Args:
            vignettes: List of vignette dictionaries
            batch_size: Number of samples to process in each batch
        """
        print(f"🚀 Starting OPTIMIZED production inference (no attention)...")
        
        # Phase 1: Generate all prompts and metadata first
        print("📋 Phase 1: Generating all prompts...")
        all_prompts = []
        all_sample_data = []  # For building final records
        
        total_permutations = self._count_total_permutations(vignettes)
        current_permutation = 0
        
        print(f"Total permutations: {total_permutations:,}")
        
        # Generate all prompts (reusing existing logic)
        for vignette_idx, vignette in enumerate(vignettes):
            print(f"  Generating prompts for vignette {vignette_idx + 1}/{len(vignettes)}: {vignette['topic']}")
            
            # Import required modules
            from field_definitions import (get_name_for_country_gender, get_pronoun, systems_to_countries_map, 
                                          safety_to_countries_map, get_verb_present_third_person, 
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
                            print(f"    Prompt generation: {current_permutation:,}/{total_permutations:,} ({progress:.1f}%)")
                        
                        # Build sample values (same logic as before)
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
                        
                        # Add pronoun and grammar helpers
                        if 'gender' in sample_values:
                            gender = sample_values['gender']
                            sample_values['pronoun'] = get_pronoun(gender)
                            sample_values['pronoun_was_were'] = get_verb_past_be(gender)
                            sample_values['pronoun_possessive'] = get_pronoun_possessive(gender)
                            # Common verb forms
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
                        
                        # Store everything
                        all_prompts.append(prompt)
                        
                        # Store data for final record construction - INCLUDE SAMPLE_ID
                        sample_data = {
                            'sample_id': current_permutation,  # CRITICAL: This is the key for proper matching
                            'meta_topic': vignette['meta_topic'],
                            'topic': vignette['topic'],
                            'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                            'vignette_text': vignette_text,
                        }
                        
                        # Add ordinal values
                        for k in ordinal_keys:
                            if f"{k}__ordinal" in sample_values:
                                sample_data[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
                        
                        all_sample_data.append(sample_data)
        
        print(f"✅ Generated {len(all_prompts):,} prompts")
        
        # Phase 2: Run batched inference (no attention)
        print(f"🔄 Phase 2: Running batched inference (batch_size={batch_size})...")
        
        try:
            all_responses = self._run_inference_batch_optimized(
                all_prompts, batch_size=batch_size
            )
        finally:
            # Always finalize (will skip if no attention collection)
            self.finalize_attention_collection()
        
        # Phase 3: Build final records
        print("📦 Phase 3: Building final records...")
        records = []
        
        for i, (sample_data, response) in enumerate(zip(all_sample_data, all_responses)):
            record = sample_data.copy()
            record['model_response'] = response
            record['inference_timestamp'] = datetime.now().isoformat()
            records.append(record)
        
        print(f"\n✅ OPTIMIZED Inference completed!")
        print(f"Total records: {len(records):,}")
        
        return records

    def _run_inference_batch_optimized(self, prompts: List[str], batch_size: int = 16) -> List[str]:
        """
        Run batched inference without attention collection.
        Optimized for speed when attention data is not needed.
        """
        all_responses = []
        
        # Process in smaller batches to manage memory
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
                ).to(self.device)
                
                # Generate responses (no attention collection)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.6,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        output_attentions=False  # Explicitly disable for speed
                    )
                
                # Process outputs - simple decoding without attention
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
                for j, prompt in enumerate(batch_prompts):
                    try:
                        response = self._run_inference(prompt)
                        all_responses.append(response)
                    except Exception as individual_error:
                        print(f"Error in individual fallback: {str(individual_error)}")
                        all_responses.append(f"ERROR: {str(individual_error)}")
        
        return all_responses


def load_attention_data(h5_path: str, metadata_path: str) -> Tuple[Dict, h5py.File]:
    """
    Load saved attention data for analysis.
    
    Returns:
        tuple: (metadata_dict, h5_file_handle)
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    h5_file = h5py.File(h5_path, 'r')
    
    return metadata, h5_file


def analyze_sample_attention(h5_file: h5py.File, sample_id: int) -> Dict:
    """
    Extract attention data for a specific sample for analysis.
    
    Args:
        h5_file: Open HDF5 file handle
        sample_id: Sample ID to analyze
        
    Returns:
        dict: Attention data for the sample
    """
    sample_group = h5_file[f'sample_{sample_id}']
    
    # Load basic info
    sample_data = {
        'sample_id': sample_id,
        'input_length': sample_group.attrs['input_length'],
        'generated_length': sample_group.attrs['generated_length'],
        'input_tokens': list(sample_group['input_tokens'][:]),
        'attention_steps': []
    }
    
    # Load attention data
    for step_name in sample_group.keys():
        if step_name.startswith('step_'):
            step_group = sample_group[step_name]
            step_data = {
                'generation_step': step_group.attrs['generation_step'],
                'layers': {}
            }
            
            for layer_name in step_group.keys():
                attention_matrix = np.array(step_group[layer_name])
                layer_idx = int(layer_name.split('_')[1])
                step_data['layers'][layer_idx] = attention_matrix
            
            sample_data['attention_steps'].append(step_data)
    
    return sample_data


def main():
    """Command line interface for production attention collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production-scale inference with optional attention collection")
    parser.add_argument("model_subdir", help="Model subdirectory or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", required=True, help="Output path for inference results")
    parser.add_argument("--attention-dir", default="attention_data",
                       help="Directory to save attention data")
    parser.add_argument("--attention-rate", type=float, default=0.8,
                       help="Fraction of samples to collect attention for (0.0-1.0)")
    parser.add_argument("--no-attention", action="store_true",
                       help="Disable attention collection (faster inference)")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Load model from Hugging Face Hub")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference (default: 8)")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ProductionAttentionPipeline(
        args.model_subdir,
        use_hf_hub=args.use_hf_hub,
        collect_attention=not args.no_attention,
        attention_sample_rate=args.attention_rate,
        storage_dir=args.attention_dir
    )
    
    # Load vignettes
    vignettes = load_vignettes(args.vignettes)
    
    # Run inference - choose method based on attention collection
    if args.no_attention:
        print("Running optimized inference without attention collection...")
        records = pipeline.generate_inference_records_optimized(vignettes, batch_size=args.batch_size)
    else:
        print("Running inference with attention collection...")
        records = pipeline.generate_inference_records_with_attention_optimized(vignettes, batch_size=args.batch_size)
    
    # Save inference results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Production run completed!")
    print(f"Inference results: {args.output}")
    if not args.no_attention:
        print(f"Attention data: {args.attention_dir}")


if __name__ == "__main__":
    main() 