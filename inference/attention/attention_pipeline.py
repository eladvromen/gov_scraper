"""
Enhanced inference pipeline with token-level attention map collection.
Focuses on preserving fine-grained attention patterns for template analysis.
"""

import json
import os
import sys
import pickle
import numpy as np
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


class TokenLevelAttentionPipeline(InferencePipeline):
    """
    Enhanced inference pipeline that captures detailed token-level attention maps.
    Designed for template analysis where word-level attention patterns are crucial.
    """
    
    def __init__(self, model_subdir, models_base_dir="models", use_hf_hub=False, 
                 collect_attention=True, attention_sample_rate=0.1):
        """
        Initialize the token-level attention pipeline.
        
        Args:
            model_subdir (str): Model subdirectory or HF Hub model name
            models_base_dir (str): Base directory for local models
            use_hf_hub (bool): Whether to use HF Hub model
            collect_attention (bool): Whether to collect attention maps
            attention_sample_rate (float): Fraction of samples to collect attention for (0.0-1.0)
        """
        super().__init__(model_subdir, models_base_dir, use_hf_hub)
        
        self.collect_attention = collect_attention
        self.attention_sample_rate = attention_sample_rate
        self.attention_data = []
        
        print(f"Token-level attention collection: {'Enabled' if collect_attention else 'Disabled'}")
        if collect_attention:
            print(f"Attention sample rate: {attention_sample_rate:.1%}")
    
    def _should_collect_attention(self, sample_idx: int, total_samples: int) -> bool:
        """Stratified sampling for attention collection."""
        if not self.collect_attention or self.attention_sample_rate <= 0:
            return False
        
        if self.attention_sample_rate >= 1.0:
            return True
        
        target_samples = int(total_samples * self.attention_sample_rate)
        if target_samples == 0:
            return False
        
        # Stratified intervals to ensure coverage across all vignettes
        interval = total_samples / target_samples
        return sample_idx % int(interval) == 0
    
    def _extract_template_tokens(self, prompt: str, vignette_text: str) -> Dict[str, List[int]]:
        """
        Identify token positions for different parts of the template.
        This helps analyze attention to specific template components.
        """
        # Tokenize the full prompt
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Find vignette text boundaries within the prompt
        vignette_start = prompt.find(vignette_text)
        if vignette_start == -1:
            return {'error': 'Vignette text not found in prompt'}
        
        # Tokenize just the vignette part for mapping
        vignette_tokens = self.tokenizer.tokenize(vignette_text)
        
        # Map template components to token positions
        template_regions = {
            'prompt_prefix': [],  # Before vignette
            'vignette_content': [],  # The vignette itself
            'prompt_suffix': [],  # After vignette (format instructions)
            'name_tokens': [],
            'demographic_tokens': [],  # age, religion, gender, country
            'narrative_tokens': [],  # action/story parts
            'legal_argument_tokens': []  # Home Office argument
        }
        
        # This is a simplified version - you might want to enhance this
        # based on your specific template structure
        prompt_before_vignette = prompt[:vignette_start]
        prompt_after_vignette = prompt[vignette_start + len(vignette_text):]
        
        prefix_tokens = self.tokenizer.tokenize(prompt_before_vignette)
        suffix_tokens = self.tokenizer.tokenize(prompt_after_vignette)
        
        # Map token positions
        current_pos = 0
        template_regions['prompt_prefix'] = list(range(current_pos, current_pos + len(prefix_tokens)))
        current_pos += len(prefix_tokens)
        
        template_regions['vignette_content'] = list(range(current_pos, current_pos + len(vignette_tokens)))
        current_pos += len(vignette_tokens)
        
        template_regions['prompt_suffix'] = list(range(current_pos, current_pos + len(suffix_tokens)))
        
        return template_regions
    
    def _run_inference_with_attention(self, prompt: str, sample_metadata: Dict) -> Tuple[str, Optional[Dict]]:
        """
        Run inference and capture detailed token-level attention weights.
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
            
            attention_data = None
            
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
            
            # Process attention weights for token-level analysis
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attention_data = self._process_token_attention(
                    outputs.attentions, inputs, outputs.sequences[0], 
                    sample_metadata, prompt
                )
            
            return response_text.strip(), attention_data
            
        except Exception as e:
            print(f"Error during inference with attention: {str(e)}")
            return f"ERROR: {str(e)}", None
    
    def _process_token_attention(self, attentions: Tuple, inputs: Dict, 
                               full_output: torch.Tensor, metadata: Dict, prompt: str) -> Dict:
        """
        Process attention weights with focus on token-level patterns for template analysis.
        """
        try:
            # Get detailed token information
            input_ids = inputs['input_ids'][0]
            input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            input_text_tokens = [(token, self.tokenizer.convert_tokens_to_string([token])) 
                                for token in input_tokens]
            
            # Get generated tokens
            generated_ids = full_output[len(input_ids):]
            generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
            
            # Extract template regions for analysis
            vignette_text = metadata.get('vignette_text', '')
            template_regions = self._extract_template_tokens(prompt, vignette_text)
            
            # Process attention for each generation step
            processed_attention = {
                'sample_id': metadata.get('sample_id'),
                'vignette_topic': metadata.get('vignette_topic'),
                'input_tokens': input_text_tokens,
                'generated_tokens': [(token, self.tokenizer.convert_tokens_to_string([token])) 
                                   for token in generated_tokens],
                'template_regions': template_regions,
                'metadata': metadata,
                'attention_details': {
                    'num_layers': len(attentions[0]) if attentions else 0,
                    'num_heads': attentions[0][0].shape[1] if attentions and len(attentions[0]) > 0 else 0,
                    'input_length': len(input_tokens),
                    'generated_length': len(generated_tokens)
                }
            }
            
            # Store attention for key generation steps
            max_steps_to_store = min(5, len(attentions))  # First 5 generation steps
            generation_attention = []
            
            for step_idx in range(max_steps_to_store):
                step_attention = attentions[step_idx]  # List of layer attentions
                
                step_data = {
                    'generation_step': step_idx,
                    'generated_token': generated_tokens[step_idx] if step_idx < len(generated_tokens) else None,
                    'layers': []
                }
                
                # Process key layers (not all to save space)
                layers_to_store = [0, len(step_attention)//4, len(step_attention)//2, 
                                 3*len(step_attention)//4, len(step_attention)-1]
                
                for layer_idx in layers_to_store:
                    if layer_idx < len(step_attention):
                        layer_attn = step_attention[layer_idx]  # [batch, heads, seq_len, seq_len]
                        
                        # Store both aggregated and individual head attention
                        layer_data = {
                            'layer_idx': layer_idx,
                            'attention_aggregated': layer_attn[0].mean(dim=0).cpu().numpy().tolist(),
                            'attention_heads': []
                        }
                        
                        # Store attention for first few heads for detailed analysis
                        num_heads_to_store = min(4, layer_attn.shape[1])
                        for head_idx in range(num_heads_to_store):
                            head_attention = layer_attn[0, head_idx].cpu().numpy()
                            
                            # Focus on attention TO input tokens FROM current generation position
                            if step_idx < head_attention.shape[0]:
                                input_attention = head_attention[len(input_ids) + step_idx, :len(input_ids)]
                                
                                layer_data['attention_heads'].append({
                                    'head_idx': head_idx,
                                    'attention_to_input': input_attention.tolist(),
                                    'top_attended_tokens': self._get_top_attended_tokens(
                                        input_attention, input_text_tokens, top_k=10
                                    )
                                })
                        
                        step_data['layers'].append(layer_data)
                
                generation_attention.append(step_data)
            
            processed_attention['generation_attention'] = generation_attention
            
            return processed_attention
            
        except Exception as e:
            print(f"Error processing token attention: {str(e)}")
            return {'error': str(e), 'metadata': metadata}
    
    def _get_top_attended_tokens(self, attention_weights: np.ndarray, 
                               tokens: List[Tuple], top_k: int = 10) -> List[Dict]:
        """Get the top-k most attended tokens with their attention scores."""
        if len(attention_weights) != len(tokens):
            return []
        
        # Get top k indices
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        
        top_tokens = []
        for idx in top_indices:
            if idx < len(tokens):
                token, text = tokens[idx]
                top_tokens.append({
                    'token_idx': int(idx),
                    'token': token,
                    'text': text,
                    'attention_score': float(attention_weights[idx])
                })
        
        return top_tokens
    
    def _run_inference(self, prompt: str, sample_metadata: Dict = None) -> str:
        """Override to potentially collect attention."""
        if self.collect_attention and sample_metadata is not None:
            # Add vignette text to metadata for template analysis
            if 'vignette_text' not in sample_metadata:
                # Extract vignette text from prompt if possible
                case_start = prompt.find("Case Details:")
                output_start = prompt.find("output format:")
                if case_start != -1 and output_start != -1:
                    sample_metadata['vignette_text'] = prompt[case_start:output_start].strip()
            
            response, attention_data = self._run_inference_with_attention(prompt, sample_metadata)
            
            if attention_data is not None:
                self.attention_data.append(attention_data)
                
            return response
        else:
            return super()._run_inference(prompt)
    
    def save_attention_data(self, output_dir: str, filename_prefix: str = "token_attention"):
        """Save token-level attention data for analysis."""
        if not self.attention_data:
            print("No attention data to save")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for inspection and analysis
        json_path = os.path.join(output_dir, f"{filename_prefix}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.attention_data, f, indent=2, ensure_ascii=False)
        
        # Save compact version without full attention matrices for overview
        compact_data = []
        for sample in self.attention_data:
            compact_sample = {
                'sample_id': sample.get('sample_id'),
                'vignette_topic': sample.get('vignette_topic'),
                'metadata': sample.get('metadata'),
                'attention_summary': []
            }
            
            for gen_step in sample.get('generation_attention', []):
                step_summary = {
                    'generation_step': gen_step['generation_step'],
                    'generated_token': gen_step.get('generated_token'),
                    'top_attended_across_layers': []
                }
                
                for layer in gen_step.get('layers', []):
                    for head in layer.get('attention_heads', []):
                        step_summary['top_attended_across_layers'].extend(
                            head.get('top_attended_tokens', [])[:3]  # Top 3 per head
                        )
                
                compact_sample['attention_summary'].append(step_summary)
            compact_data.append(compact_sample)
        
        compact_path = os.path.join(output_dir, f"{filename_prefix}_summary.json")
        with open(compact_path, 'w', encoding='utf-8') as f:
            json.dump(compact_data, f, indent=2, ensure_ascii=False)
        
        # Create analysis-ready CSV for easy exploration
        self._create_attention_csv(output_dir, filename_prefix)
        
        print(f"Token-level attention data saved:")
        print(f"  Full data: {json_path}")
        print(f"  Summary: {compact_path}")
        print(f"  Total samples: {len(self.attention_data)}")
    
    def _create_attention_csv(self, output_dir: str, filename_prefix: str):
        """Create CSV files for easy attention analysis."""
        try:
            import pandas as pd
            
            # Flatten attention data for analysis
            attention_records = []
            
            for sample in self.attention_data:
                sample_id = sample.get('sample_id')
                topic = sample.get('vignette_topic')
                
                for gen_step in sample.get('generation_attention', []):
                    step_idx = gen_step['generation_step']
                    gen_token = gen_step.get('generated_token')
                    
                    for layer in gen_step.get('layers', []):
                        layer_idx = layer['layer_idx']
                        
                        for head in layer.get('attention_heads', []):
                            head_idx = head['head_idx']
                            
                            for token_info in head.get('top_attended_tokens', [])[:5]:  # Top 5
                                attention_records.append({
                                    'sample_id': sample_id,
                                    'vignette_topic': topic,
                                    'generation_step': step_idx,
                                    'generated_token': gen_token,
                                    'layer_idx': layer_idx,
                                    'head_idx': head_idx,
                                    'attended_token': token_info['token'],
                                    'attended_text': token_info['text'],
                                    'attention_score': token_info['attention_score'],
                                    'token_position': token_info['token_idx']
                                })
            
            if attention_records:
                df = pd.DataFrame(attention_records)
                csv_path = os.path.join(output_dir, f"{filename_prefix}_analysis.csv")
                df.to_csv(csv_path, index=False)
                print(f"  Analysis CSV: {csv_path}")
                
        except ImportError:
            print("  pandas not available, skipping CSV creation")


# Integration with existing pipeline
def create_attention_pipeline_wrapper(model_subdir, attention_sample_rate=0.1, **kwargs):
    """
    Factory function to create attention pipeline with same interface as base pipeline.
    """
    return TokenLevelAttentionPipeline(
        model_subdir, 
        collect_attention=True,
        attention_sample_rate=attention_sample_rate,
        **kwargs
    )


def main():
    """Command line interface for token-level attention inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with token-level attention collection")
    parser.add_argument("model_subdir", help="Model subdirectory or HF Hub model name")
    parser.add_argument("--vignettes", default="vignettes/complete_vignettes.json",
                       help="Path to vignettes JSON file")
    parser.add_argument("--output", required=True, help="Output path for inference results")
    parser.add_argument("--attention-dir", help="Directory to save attention data")
    parser.add_argument("--attention-rate", type=float, default=0.1,
                       help="Fraction of samples to collect attention for (0.0-1.0)")
    parser.add_argument("--no-attention", action="store_true",
                       help="Disable attention collection")
    parser.add_argument("--use-hf-hub", action="store_true",
                       help="Load model from Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TokenLevelAttentionPipeline(
        args.model_subdir,
        use_hf_hub=args.use_hf_hub,
        collect_attention=not args.no_attention,
        attention_sample_rate=args.attention_rate
    )
    
    # Run inference with attention collection
    print("Starting token-level attention inference...")
    
    # Load vignettes and run inference
    vignettes = load_vignettes(args.vignettes)
    
    # For demonstration, run on a small subset first
    sample_vignettes = vignettes[:2]  # First 2 vignettes for testing
    
    records = []
    total_perms = pipeline._count_total_permutations(sample_vignettes)
    current_perm = 0
    
    for vignette in sample_vignettes:
        # Generate a few samples per vignette
        sample_records = pipeline.generate_samples([vignette], num_samples=5)
        
        for record in sample_records:
            current_perm += 1
            should_collect = pipeline._should_collect_attention(current_perm, total_perms)
            
            if should_collect:
                print(f"Collecting attention for sample {current_perm}")
                
                sample_metadata = {
                    'sample_id': current_perm,
                    'vignette_topic': record['topic'],
                    'meta_topic': record['meta_topic'],
                    'should_collect': True
                }
                
                prompt = pipeline._create_prompt(record['vignette_text'])
                response = pipeline._run_inference(prompt, sample_metadata)
                record['model_response'] = response
                record['attention_collected'] = True
            else:
                prompt = pipeline._create_prompt(record['vignette_text'])
                response = pipeline._run_inference(prompt)
                record['model_response'] = response
                record['attention_collected'] = False
            
            records.append(record)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    
    # Save attention data if collected and directory specified
    if args.attention_dir and pipeline.attention_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_subdir)
        prefix = f"attention_{model_name}_{timestamp}"
        pipeline.save_attention_data(args.attention_dir, prefix)
    
    print(f"Completed! Generated {len(records)} samples")
    print(f"Collected attention for {len(pipeline.attention_data)} samples")


if __name__ == "__main__":
    main() 