#!/usr/bin/env python3
"""
Example usage script for vignette_viewer.py

This script demonstrates common usage patterns for the HTML vignette viewer.
"""

import subprocess
import sys
from pathlib import Path

def run_viewer(topics=None, output_filename=None):
    """Run the vignette viewer with specified topics."""
    cmd = [sys.executable, "vignette_viewer.py"]
    
    if topics:
        cmd.extend(["--topics"] + topics)
    
    if output_filename:
        cmd.extend(["--output", output_filename])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Generated successfully: {output_filename or 'vignette_viewer.html'}")
            print(result.stdout)
        else:
            print(f"✗ Error generating report:")
            print(result.stderr)
    except Exception as e:
        print(f"Error running viewer: {e}")

def main():
    print("Vignette Viewer - Example Usage")
    print("=" * 50)
    
    examples = [
        {
            "name": "All Contradiction Cases",
            "topics": [
                "Contradiction: Dates of persecution",
                "Contradiction: Family involvement in the persecution",
                "Contradiction: Location of harm",
                "Contradiction: Persecutor identity confusion",
                "Contradiction: Sequence of events"
            ],
            "filename": "contradiction_cases.html"
        },
        {
            "name": "All Disclosure Cases", 
            "topics": [
                "Disclosure: Domestic violence & criminal threats",
                "Disclosure: Ethnic violence & family separation",
                "Disclosure: Persecution for sexual orientation & mental health crisis",
                "Disclosure: Political persecution & sexual violence",
                "Disclosure: Religious persecution & mental health"
            ],
            "filename": "disclosure_cases.html"
        },
        {
            "name": "Third Safe Country Cases",
            "topics": [
                "3rd safe country - Asylum seeker circumstances",
                "3rd safe country - Connections to country B",
                "3rd safe country - Country safety definition",
                "3rd safe country – systemic condition"
            ],
            "filename": "safe_country_cases.html"
        },
        {
            "name": "Intentions Cases",
            "topics": [
                "Intentions regarding education in the UK",
                "Intentions regarding work in the UK"
            ],
            "filename": "intentions_cases.html"
        },
        {
            "name": "Cases with Biggest Decision Differences",
            "topics": [
                "Disclosure: Political persecution & sexual violence",
                "Contradiction: Sequence of events",
                "Intentions regarding education in the UK"
            ],
            "filename": "high_difference_cases.html"
        }
    ]
    
    print("Available example reports:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']} ({len(example['topics'])} topics)")
    
    print(f"{len(examples)+1}. All cases (no filter)")
    print(f"{len(examples)+2}. Custom topic selection")
    print("0. Exit")
    
    while True:
        try:
            choice = input(f"\nSelect option (0-{len(examples)+2}): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                break
            elif choice == str(len(examples)+1):
                print("\nGenerating report for all cases...")
                run_viewer(topics=None, output_filename="all_cases.html")
            elif choice == str(len(examples)+2):
                print("\nEnter topics (one per line, empty line to finish):")
                custom_topics = []
                while True:
                    topic = input("Topic: ").strip()
                    if not topic:
                        break
                    custom_topics.append(topic)
                
                if custom_topics:
                    filename = input("Output filename (default: custom_cases.html): ").strip()
                    if not filename:
                        filename = "custom_cases.html"
                    print(f"\nGenerating report for {len(custom_topics)} custom topics...")
                    run_viewer(topics=custom_topics, output_filename=filename)
                else:
                    print("No topics specified.")
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(examples):
                    example = examples[choice_idx]
                    print(f"\nGenerating: {example['name']}")
                    print(f"Topics: {', '.join(example['topics'])}")
                    run_viewer(topics=example['topics'], output_filename=example['filename'])
                else:
                    print("Invalid choice. Please try again.")
        
        except (ValueError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main() 