#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
"""
Case Outcome Prediction Pipeline Runner
======================================

This script runs the complete pipeline for case outcome prediction:
1. Data preprocessing
2. Train/test split
3. Baseline evaluation
4. Fine-tuning
5. Final evaluation
6. Inference

Usage:
    python run_pipeline.py [--steps STEPS] [--skip-inference]
    
    --steps: Comma-separated list of steps to run (1,2,3,4,5,6)
    --skip-inference: Skip the inference step (useful for testing)
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
import time

class PipelineRunner:
    def __init__(self):
        self.steps = {
            1: ("Data Preprocessing", "01_data_preprocessing.py"),
            2: ("Train/Test Split", "02_train_test_split.py"),
            3: ("Baseline Evaluation", "03_baseline_evaluation.py"),
            4: ("Fine-tuning", "04_fine_tuning.py"),
            5: ("Final Evaluation", "05_final_evaluation.py"),
            6: ("Inference Pipeline", "06_inference_pipeline.py")
        }
        
        self.start_time = None
        self.step_times = {}
        
    def run_step(self, step_num):
        """Run a single pipeline step"""
        if step_num not in self.steps:
            print(f"❌ Invalid step number: {step_num}")
            return False
            
        step_name, script_name = self.steps[step_num]
        
        print(f"\n{'='*60}")
        print(f"STEP {step_num}: {step_name.upper()}")
        print(f"{'='*60}")
        print(f"Running: {script_name}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        step_start = time.time()
        
        try:
            # Run the script
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=False, 
                                  text=True, 
                                  check=True)
            
            step_end = time.time()
            step_duration = step_end - step_start
            self.step_times[step_num] = step_duration
            
            print(f"\n✅ Step {step_num} completed successfully in {step_duration:.1f} seconds")
            return True
            
        except subprocess.CalledProcessError as e:
            step_end = time.time()
            step_duration = step_end - step_start
            
            print(f"\n❌ Step {step_num} failed after {step_duration:.1f} seconds")
            print(f"Error: {e}")
            return False
            
        except FileNotFoundError:
            print(f"\n❌ Script not found: {script_name}")
            print("Make sure all pipeline scripts are in the current directory")
            return False
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Step {step_num} interrupted by user")
            return False
    
    def check_prerequisites(self):
        """Check if all required files exist"""
        print("Checking prerequisites...")
        
        # Check if all scripts exist
        missing_scripts = []
        for step_num, (step_name, script_name) in self.steps.items():
            if not os.path.exists(script_name):
                missing_scripts.append(script_name)
        
        if missing_scripts:
            print("❌ Missing required scripts:")
            for script in missing_scripts:
                print(f"  - {script}")
            return False
        
        # Check if data files exist
        required_data_files = [
            "/data/shil6369/gov_scraper/data/leaglBERT_training.pkl",
            "/data/shil6369/gov_scraper/data/processed_legal_cases.parquet"
        ]
        
        missing_data = []
        for data_file in required_data_files:
            if not os.path.exists(data_file):
                missing_data.append(data_file)
        
        if missing_data:
            print("❌ Missing required data files:")
            for data_file in missing_data:
                print(f"  - {data_file}")
            return False
        
        print("✅ All prerequisites met")
        return True
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline steps"""
        print("\n" + "="*60)
        print("CASE OUTCOME PREDICTION PIPELINE")
        print("="*60)
        print("This pipeline will:")
        print("1. Preprocess training and inference data")
        print("2. Create stratified train/validation/test splits")
        print("3. Evaluate baseline LegalBERT performance")
        print("4. Fine-tune LegalBERT on case outcome data")
        print("5. Evaluate fine-tuned model performance")
        print("6. Run inference on new legal cases")
        print()
        print("Expected duration: 30-60 minutes (depending on hardware)")
        print("Requirements: GPU recommended for steps 3-6")
    
    def run_pipeline(self, steps_to_run=None, skip_inference=False):
        """Run the complete pipeline"""
        self.start_time = time.time()
        
        # Print summary
        self.print_pipeline_summary()
        
        # Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Determine which steps to run
        if steps_to_run is None:
            steps_to_run = list(self.steps.keys())
        
        if skip_inference and 6 in steps_to_run:
            steps_to_run.remove(6)
            print("⚠️  Skipping inference step as requested")
        
        print(f"\nRunning steps: {steps_to_run}")
        input("\nPress Enter to continue or Ctrl+C to cancel...")
        
        # Run each step
        failed_steps = []
        for step_num in steps_to_run:
            success = self.run_step(step_num)
            if not success:
                failed_steps.append(step_num)
                print(f"\n❌ Pipeline stopped at step {step_num}")
                break
        
        # Print final summary
        self.print_final_summary(steps_to_run, failed_steps)
        
        return len(failed_steps) == 0
    
    def print_final_summary(self, attempted_steps, failed_steps):
        """Print final pipeline summary"""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        
        print(f"\nStep execution times:")
        for step_num in attempted_steps:
            if step_num in self.step_times:
                step_name = self.steps[step_num][0]
                duration = self.step_times[step_num]
                status = "✅ COMPLETED" if step_num not in failed_steps else "❌ FAILED"
                print(f"  Step {step_num} ({step_name}): {duration:.1f}s - {status}")
        
        if failed_steps:
            print(f"\n❌ Pipeline failed at step(s): {failed_steps}")
            print("Check the error messages above for details")
        else:
            print(f"\n✅ Pipeline completed successfully!")
            print("Check the generated files for results and analysis")
            
            # List key output files
            output_files = [
                "processed_training_data.parquet",
                "processed_inference_data.parquet",
                "train_data.parquet",
                "val_data.parquet", 
                "test_data.parquet",
                "baseline_results.json",
                "fine_tuned_legal_bert_final/",
                "final_evaluation_results.json",
                "case_outcome_predictions.parquet"
            ]
            
            existing_files = [f for f in output_files if os.path.exists(f)]
            if existing_files:
                print(f"\nKey output files:")
                for file in existing_files:
                    print(f"  - {file}")

def main():
    parser = argparse.ArgumentParser(description="Run the case outcome prediction pipeline")
    parser.add_argument("--steps", type=str, help="Comma-separated list of steps to run (e.g., '1,2,3')")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference step")
    
    args = parser.parse_args()
    
    # Parse steps if provided
    steps_to_run = None
    if args.steps:
        try:
            steps_to_run = [int(s.strip()) for s in args.steps.split(',')]
            print(f"Running only steps: {steps_to_run}")
        except ValueError:
            print("❌ Invalid steps format. Use comma-separated numbers (e.g., '1,2,3')")
            return 1
    
    # Create and run pipeline
    runner = PipelineRunner()
    success = runner.run_pipeline(steps_to_run, args.skip_inference)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 