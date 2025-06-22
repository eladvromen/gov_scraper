import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
import random

# Try to import torch and transformers
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add vignettes directory to path for imports
vignettes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../vignettes'))
sys.path.insert(0, vignettes_path)

from field_definitions import * 

# Import existing vignette processing functions
from utils import load_vignettes, resolve_field_reference

class InferencePipeline:
    def __init__(self, model_subdir, models_base_dir="models", use_hf_hub=False):
        """
        Initialize the inference pipeline.
        
        Args:
            model_subdir (str): Subdirectory name under models_base_dir OR HF Hub model name
            models_base_dir (str): Base directory containing model subdirectories  
            use_hf_hub (bool): If True, treat model_subdir as HF Hub model name
        """
        self.models_base_dir = models_base_dir
        self.model_subdir = model_subdir
        self.use_hf_hub = use_hf_hub
        
        if use_hf_hub:
            self.model_path = model_subdir  # Use HF Hub model name directly
            print(f"Using Hugging Face Hub model: {self.model_path}")
        else:
            self.model_path = os.path.join(models_base_dir, model_subdir)
            # Check if local model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model directory not found: {self.model_path}")
            print(f"Using local model: {self.model_path}")
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_prompt = None  # Track the last prompt used
        
        print(f"Using device: {self.device}")
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer from the specified directory."""
        print(f"Loading model from: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Optimized model loading for A100s
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=True,  # Enable KV caching for faster generation
                low_cpu_mem_usage=True,
                max_memory={0: "70GB", 1: "70GB", 2: "40GB", 3: "40GB"}  # Better GPU memory allocation
            )
            
            # Ensure pad token is set and configure for decoder-only models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set left padding for decoder-only models (like Llama)
            self.tokenizer.padding_side = 'left'
            
            # Optimize model with PyTorch 2.0 compilation (if available)
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("Model compiled with PyTorch 2.0 optimizations")
            except Exception as compile_error:
                print(f"PyTorch compilation not available: {compile_error}")
                
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def generate_inference_records(self, vignettes):
        """
        Generate inference records for all vignette permutations.
        This extends the existing generate_analytics_records functionality.
        
        Args:
            vignettes (list): List of vignette configurations
            
        Returns:
            list: List of records with model responses
        """
        records = []
        total_permutations = self._count_total_permutations(vignettes)
        current_permutation = 0
        
        print(f"Generating inference for {total_permutations} total permutations...")
        
        for vignette in vignettes:
            print(f"Processing vignette: {vignette['topic']}")
            
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

            # Generate all permutations for this vignette
            for generic_vals in product(*generic_lists):
                for ordinal_vals in product(*ordinal_lists) if ordinal_lists else [()]:
                    for horizontal_vals in product(*horizontal_lists) if horizontal_lists else [()]:
                        current_permutation += 1
                        
                        # Progress indicator
                        if current_permutation % 100 == 0:
                            print(f"  Progress: {current_permutation}/{total_permutations} ({current_permutation/total_permutations*100:.1f}%)")
                        
                        sample_values = {}
                        
                        # Fill generic fields
                        for k, v in zip(generic_keys, generic_vals):
                            sample_values[k] = v
                            
                        # Fill ordinal fields (label and value)
                        for k, v in zip(ordinal_keys, ordinal_vals):
                            sample_values[k] = v
                            sample_values[f"{k}__ordinal"] = ordinal_fields[k][v]
                            
                        # Fill horizontal fields
                        for k, v in zip(horizontal_keys, horizontal_vals):
                            sample_values[k] = v
                            
                        # Handle derived fields
                        if derived_fields:
                            for dfield, dspec in derived_fields.items():
                                if dfield == "name" and "country" in sample_values and "gender" in sample_values:
                                    sample_values["name"] = get_name_for_country_gender(sample_values["country"], sample_values["gender"])
                                elif dfield == "country_B":
                                    # Use mapping to get possible country_B values
                                    mapping = dspec["mapping"]
                                    source_field = dspec["source_field"]
                                    if mapping == "systems_to_countries_map":
                                        val = sample_values.get(source_field)
                                        if val and val in systems_to_countries_map:
                                            sample_values["country_B"] = systems_to_countries_map[val][0]  # pick first for determinism
                                    elif mapping == "safety_to_countries_map":
                                        val = sample_values.get(source_field)
                                        if val and val in safety_to_countries_map:
                                            sample_values["country_B"] = safety_to_countries_map[val][0]
                        
                        # Add pronoun
                        if 'gender' in sample_values:
                            sample_values['pronoun'] = get_pronoun(sample_values['gender'])
                        
                        # Generate vignette text
                        try:
                            vignette_text = vignette["vignette_template"].format(**sample_values)
                        except KeyError as e:
                            print(f"Warning: Missing field {e} for vignette template")
                            continue
                        
                        # Create prompt for the model
                        prompt = self._create_prompt(vignette_text)
                        
                        # Run inference
                        model_response = self._run_inference(prompt)
                        
                        # Build analytics record
                        record = {
                            'meta_topic': vignette['meta_topic'],
                            'topic': vignette['topic'],
                            'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                            'vignette_text': vignette_text,
                            'model_response': model_response,
                            'prompt_used': prompt,
                            'inference_timestamp': datetime.now().isoformat()
                        }
                        
                        # Add ordinal values as separate fields
                        for k in ordinal_keys:
                            if f"{k}__ordinal" in sample_values:
                                record[f"fields.{k}__ordinal"] = sample_values[f"{k}__ordinal"]
                        
                        records.append(record)
        
        print(f"Completed inference for {len(records)} permutations")
        return records
    
    def _count_total_permutations(self, vignettes):
        """Count total number of permutations across all vignettes."""
        total = 0
        for vignette in vignettes:
            generic_fields = vignette.get('generic_fields', {})
            ordinal_fields = vignette.get('ordinal_fields', {})
            horizontal_fields = vignette.get('horizontal_fields', {})
            
            generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_fields.keys()]
            ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_fields.keys()]
            horizontal_lists = [horizontal_fields[k] for k in horizontal_fields.keys()]
            
            generic_count = 1
            for lst in generic_lists:
                generic_count *= len(lst)
                
            ordinal_count = 1
            for lst in ordinal_lists:
                ordinal_count *= len(lst)
                
            horizontal_count = 1
            for lst in horizontal_lists:
                horizontal_count *= len(lst)
                
            total += generic_count * ordinal_count * horizontal_count
        
        return total
    
    def _create_prompt(self, vignette_text):
        """
        Create a prompt for the model using the vignette text.
        This uses the meta_prompt from field_definitions.py
        
        Args:
            vignette_text (str): The generated vignette text
            
        Returns:
            str: The formatted prompt for the model
        """
        return meta_prompt.format(vignette_text)
    
    def _run_inference(self, prompt):
        """
        Run inference on the given prompt.
        
        Args:
            prompt (str): The input prompt
            
        Returns:
            str: The model's response
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
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,          # Much shorter to prevent runaway
                    do_sample=True,
                    temperature=0.3,             # More focused
                    top_p=0.7,
                    repetition_penalty= 1.2,      # Reduce repetition
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response (exclude the input prompt)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return f"ERROR: {str(e)}"

    def _run_inference_batch(self, prompts, batch_size=16):
        """
        Run inference on a batch of prompts for improved efficiency.
        
        Args:
            prompts (list): List of input prompts
            batch_size (int): Number of prompts to process in each batch
            
        Returns:
            list: List of model responses
        """
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    padding=True
                ).to(self.device)
                
                # Generate responses with optimized parameters
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        do_sample=True,
                        temperature=0.5,#0.7
                        top_p=0.6,
                        repetition_penalty=1.2,#1.1
                        pad_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,  # Enable KV caching
                        num_beams=1  # Ensure no beam search for speed
                    )
                
                # Decode responses
                batch_responses = []
                for j, output in enumerate(outputs):
                    input_length = inputs['input_ids'][j].shape[0]
                    response_tokens = output[input_length:]
                    response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    batch_responses.append(response.strip())
                
                all_responses.extend(batch_responses)
                
                # Progress indicator for large batches
                if len(prompts) > batch_size * 2:
                    processed = min(i + batch_size, len(prompts))
                    print(f"    Batch progress: {processed}/{len(prompts)} prompts processed")
                
                # Clear GPU cache between batches for better memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error during batch inference: {str(e)}")
                # Fallback to individual processing for this batch
                for prompt in batch:
                    try:
                        response = self._run_inference(prompt)
                        all_responses.append(response)
                    except Exception as individual_error:
                        print(f"Error in individual fallback: {str(individual_error)}")
                        all_responses.append(f"ERROR: {str(individual_error)}")
        
        return all_responses
    
    def get_last_prompt(self):
        """Get the last prompt used for inference."""
        return self.last_prompt

    def run_inference(self, vignette_text):
        """
        Run inference on a single vignette text.
        
        Args:
            vignette_text (str): The vignette text to run inference on
            
        Returns:
            str: The model's response
        """
        prompt = self._create_prompt(vignette_text)
        self.last_prompt = prompt  # Track the prompt
        return self._run_inference(prompt)

    def run_inference_auto(self, vignette_texts, batch_size=8):
        """
        Run inference on vignette text(s) - automatically handles single or batch processing.
        
        Args:
            vignette_texts (str or list): Single vignette text or list of vignette texts
            batch_size (int): Batch size for processing (ignored if single text)
            
        Returns:
            str or list: Single response or list of responses
        """
        if isinstance(vignette_texts, str):
            # Single text - use individual processing
            return self.run_inference(vignette_texts)
        else:
            # List of texts - use batch processing
            prompts = [self._create_prompt(text) for text in vignette_texts]
            return self._run_inference_batch(prompts, batch_size)

    def run_pipeline(self, vignettes_path, output_path):
        """
        Run the complete inference pipeline.
        
        Args:
            vignettes_path (str): Path to vignettes JSON file
            output_path (str): Path to save inference results
        """
        print(f"Starting inference pipeline...")
        print(f"Vignettes: {vignettes_path}")
        print(f"Output: {output_path}")
        print(f"Model: {self.model_path}")
        
        # Load vignettes
        vignettes = load_vignettes(vignettes_path)
        print(f"Loaded {len(vignettes)} vignettes")
        
        # Generate inference records
        inference_records = self.generate_inference_records(vignettes)
        
        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(inference_records, f, indent=2, ensure_ascii=False)
        
        print(f"Inference complete! Results saved to: {output_path}")
        print(f"Total records generated: {len(inference_records)}")
        
        return inference_records

    def generate_samples(self, vignettes, num_samples=None, random_seed=42):
        """
        Generate samples from vignettes without running inference.
        
        Args:
            vignettes (list): List of vignette dictionaries
            num_samples (int): Number of samples to generate (None for all permutations)
            random_seed (int): Random seed for sampling
            
        Returns:
            list: List of sample dictionaries with vignette_text and fields
        """
        random.seed(random_seed)
        
        samples = []
        
        for vignette in vignettes:
            # Extract field configurations
            generic_fields = vignette.get('generic_fields', {})
            ordinal_fields = vignette.get('ordinal_fields', {})
            horizontal_fields = vignette.get('horizontal_fields', {})
            derived_fields = vignette.get('derived_fields', {})
            
            # Get field keys and lists
            generic_keys = list(generic_fields.keys())
            ordinal_keys = list(ordinal_fields.keys())
            horizontal_keys = list(horizontal_fields.keys())
            
            generic_lists = [resolve_field_reference(generic_fields[k]) for k in generic_keys]
            ordinal_lists = [list(ordinal_fields[k].keys()) for k in ordinal_keys]
            horizontal_lists = [horizontal_fields[k] for k in horizontal_keys]
            
            # Generate all permutations for this vignette
            all_combinations = list(itertools.product(*(generic_lists + ordinal_lists + horizontal_lists)))
            
            # Sample if requested
            if num_samples and len(all_combinations) > num_samples:
                sampled_combinations = random.sample(all_combinations, num_samples)
            else:
                sampled_combinations = all_combinations
            
            # Generate samples
            for combination in sampled_combinations:
                sample_values = {}
                
                # Split combination back into components
                generic_vals = combination[:len(generic_keys)]
                ordinal_vals = combination[len(generic_keys):len(generic_keys)+len(ordinal_keys)]
                horizontal_vals = combination[len(generic_keys)+len(ordinal_keys):]
                
                # Fill generic fields
                for k, v in zip(generic_keys, generic_vals):
                    sample_values[k] = v
                
                # Fill ordinal fields (label and value)
                for k, v in zip(ordinal_keys, ordinal_vals):
                    sample_values[k] = v
                    sample_values[f"{k}__ordinal"] = ordinal_fields[k][v]
                    
                # Fill horizontal fields
                for k, v in zip(horizontal_keys, horizontal_vals):
                    sample_values[k] = v
                    
                # Handle derived fields
                if derived_fields:
                    for dfield, dspec in derived_fields.items():
                        if dfield == "name" and "country" in sample_values and "gender" in sample_values:
                            sample_values["name"] = get_name_for_country_gender(sample_values["country"], sample_values["gender"])
                        elif dfield == "country_B":
                            # Use mapping to get possible country_B values
                            mapping = dspec["mapping"]
                            source_field = dspec["source_field"]
                            if mapping == "systems_to_countries_map":
                                val = sample_values.get(source_field)
                                if val and val in systems_to_countries_map:
                                    sample_values["country_B"] = systems_to_countries_map[val][0]  # pick first for determinism
                            elif mapping == "safety_to_countries_map":
                                val = sample_values.get(source_field)
                                if val and val in safety_to_countries_map:
                                    sample_values["country_B"] = safety_to_countries_map[val][0]
                
                # Add pronoun
                if 'gender' in sample_values:
                    sample_values['pronoun'] = get_pronoun(sample_values['gender'])
                
                # Generate vignette text
                try:
                    vignette_text = vignette["vignette_template"].format(**sample_values)
                    
                    sample = {
                        'meta_topic': vignette['meta_topic'],
                        'topic': vignette['topic'],
                        'fields': {k: sample_values.get(k) for k in list(generic_keys) + list(ordinal_keys) + list(horizontal_keys) + list(derived_fields.keys()) if k in sample_values},
                        'vignette_text': vignette_text
                    }
                    samples.append(sample)
                    
                except KeyError as e:
                    print(f"Warning: Missing field {e} for vignette template")
                    continue
        
        return samples

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Run inference on vignette permutations")
    parser.add_argument("model_subdir", help="Model subdirectory under models/")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json", 
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", default="inference/inference_results.json",
                       help="Output path for inference results")
    parser.add_argument("--models-dir", default="models",
                       help="Base directory containing model subdirectories")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = InferencePipeline(args.model_subdir, args.models_dir)
    
    # Run inference
    pipeline.run_pipeline(args.vignettes, args.output)

if __name__ == "__main__":
    main() 