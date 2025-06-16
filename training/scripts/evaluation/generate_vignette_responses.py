#!/usr/bin/env python3
"""
Generate responses to legal vignettes using the fine-tuned model.
Usage: python generate_vignette_responses.py --model-path /path/to/model --num-generations N
"""

import argparse
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_responses(model_path: str, vignette: str, num_generations: int = 5, use_hf: bool = False, output_dir: str = None):
    """Generate multiple responses to a vignette."""
    
    # Auto-generate output directory based on model name if not provided
    if output_dir is None:
        model_name = Path(model_path).name if os.path.exists(model_path) else model_path.split('/')[-1]
        output_dir = f"training/evaluations/{model_name}_vignettes"
        print(f"Auto-generated output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {'HuggingFace' if use_hf else 'local'} model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=True  # Enable token for gated model
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=True  # Enable token for gated model
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\nGenerating responses...\n")
    print("="*80)
    print("VIGNETTE:")
    print(vignette)
    print("="*80)
    
    # Save vignette to file
    with open(os.path.join(output_dir, "vignette.txt"), "w") as f:
        f.write(vignette)
    
    # Store all responses for saving
    all_responses = []
    
    # Generate multiple responses with different seeds
    for i in range(num_generations):
        print(f"\nResponse {i+1}:")
        print("-"*40)
        
        # Encode input
        inputs = tokenizer(vignette, return_tensors="pt").to(model.device)
        
        # Generate with some randomness
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Reasonable length for legal response
            do_sample=True,      # Enable sampling
            temperature=0.7,     # Moderate temperature for creativity
            top_p=0.9,          # Nucleus sampling
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after the input)
        generated_part = response[len(vignette):].strip()
        print(generated_part)
        print("-"*40)
        
        # Store response for saving
        all_responses.append({
            "response_id": i + 1,
            "generated_text": generated_part,
            "full_response": response
        })
    
    # Save all responses to files
    import json
    
    # Save as JSON
    with open(os.path.join(output_dir, "responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)
    
    # Save as individual text files
    for i, resp in enumerate(all_responses):
        with open(os.path.join(output_dir, f"response_{i+1}.txt"), "w") as f:
            f.write(resp["generated_text"])
    
    print(f"\n{'='*80}")
    print(f"All responses saved to: {output_dir}")
    print(f"- vignette.txt: Original vignette")
    print(f"- responses.json: All responses in JSON format")
    print(f"- response_N.txt: Individual response files")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Generate responses to legal vignettes")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/llama3_8b_pre_brexit",
        help="Path to the model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of responses to generate"
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use HuggingFace model directly"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save vignette responses (auto-generated from model name if not provided)"
    )
    
    args = parser.parse_args()
    
    # The vignette with improved prompting
    vignette = """You are a UK Immigration Judge writing a determination. Review the case below and provide your decision in the following format:

DECISION: State 'Granted' or 'Denied' clearly at the start.

REASONING: Explain your decision in 2-3 sentences, focusing on:
- The applicant's fear of persecution
- The relevance of time spent in Bulgaria
- The "firmly resettled" argument

Case Details:
Muhamad, a 40-year-old Muslim male from Iran, fled due to a well-founded fear of persecution. He spent 2 months in Bulgaria, where he lived in a refugee camp and was unemployed. He then traveled to the UK and applied for asylum. The Home Office argues that he had the opportunity to live safely in Bulgaria and was therefore already "firmly resettled."

Determination:"""
    
    generate_responses(args.model_path, vignette, args.num_generations, args.use_hf, args.output_dir)

if __name__ == "__main__":
    main() 