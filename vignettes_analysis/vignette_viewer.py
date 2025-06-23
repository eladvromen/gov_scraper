import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib

def load_processed_results(file_path):
    """Load processed inference results and create lookup dictionaries"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data['processed_results']
    
    # Create sample_id-based lookup
    sample_lookup = {}
    missing_sample_id_count = 0
    
    for result in results:
        sample_id = result.get('sample_id')
        if sample_id is None:
            missing_sample_id_count += 1
            print(f"Warning: Result missing sample_id: {result.get('metadata', {}).get('topic', 'Unknown')}")
            continue
            
        sample_lookup[sample_id] = result
    
    if missing_sample_id_count > 0:
        print(f"Warning: {missing_sample_id_count} results missing sample_id")
    
    print(f"Loaded {len(sample_lookup)} results with sample_id from {file_path}")
    
    # Create topic-based lookup for filtering
    topic_lookup = {}
    for sample_id, result in sample_lookup.items():
        topic = result.get('metadata', {}).get('topic', 'Unknown')
        if topic not in topic_lookup:
            topic_lookup[topic] = []
        topic_lookup[topic].append(sample_id)
    
    return sample_lookup, topic_lookup

def find_matching_samples(pre_brexit_samples, post_brexit_samples, target_topics=None):
    """Find samples that exist in both datasets using sample_id"""
    
    # Get common sample IDs
    pre_brexit_ids = set(pre_brexit_samples.keys())
    post_brexit_ids = set(post_brexit_samples.keys())
    common_sample_ids = pre_brexit_ids.intersection(post_brexit_ids)
    
    print(f"Pre-Brexit samples: {len(pre_brexit_ids)}")
    print(f"Post-Brexit samples: {len(post_brexit_ids)}")
    print(f"Common sample IDs: {len(common_sample_ids)}")
    
    if len(common_sample_ids) == 0:
        print("ERROR: No matching sample IDs found!")
        return []
    
    # Filter by topics if specified
    matching_samples = []
    
    for sample_id in common_sample_ids:
        pre_result = pre_brexit_samples[sample_id]
        post_result = post_brexit_samples[sample_id]
        
        topic = pre_result.get('metadata', {}).get('topic', 'Unknown')
        
        # Filter by target topics if specified
        if target_topics and topic not in target_topics:
            continue
        
        # Create matching sample entry
        sample_match = {
            'sample_id': sample_id,
            'topic': topic,
            'meta_topic': pre_result.get('metadata', {}).get('meta_topic', 'Unknown'),
            'fields': pre_result.get('metadata', {}).get('fields', {}),
            'vignette_text': pre_result.get('metadata', {}).get('vignette_text', ''),
            'pre_brexit': pre_result,
            'post_brexit': post_result,
            'decisions_match': pre_result.get('decision') == post_result.get('decision')
        }
        
        matching_samples.append(sample_match)
    
    return matching_samples

def generate_html_report(cases: List[Dict], topics: List[str], output_file: str):
    """Generate HTML report for manual evaluation."""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vignette Analysis - Manual Evaluation</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #e0e0e0;
            }}
            .header h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            .summary {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 30px;
            }}
            .case-card {{
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-bottom: 30px;
                overflow: hidden;
                background: white;
            }}
            .case-header {{
                background: #34495e;
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .case-title {{
                font-size: 1.2em;
                font-weight: bold;
            }}
            .case-details {{
                background: #ecf0f1;
                padding: 10px 15px;
                font-size: 0.9em;
                border-bottom: 1px solid #bdc3c7;
            }}
            .vignette-text {{
                background: #f8f9fa;
                padding: 20px;
                border-left: 4px solid #3498db;
                margin: 0;
                font-style: italic;
                line-height: 1.8;
            }}
            .decisions-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0;
            }}
            .decision-panel {{
                padding: 20px;
                border-right: 1px solid #e0e0e0;
            }}
            .decision-panel:last-child {{
                border-right: none;
            }}
            .decision-panel h3 {{
                margin-top: 0;
                padding-bottom: 10px;
                border-bottom: 2px solid #e0e0e0;
            }}
            .pre-brexit {{
                background: #fdf2e9;
            }}
            .post-brexit {{
                background: #e8f5e8;
            }}
            .decision {{
                font-size: 1.1em;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                text-align: center;
            }}
            .decision.granted {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .decision.denied {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .reasoning {{
                background: white;
                padding: 15px;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
                margin-top: 10px;
            }}
            .match-indicator {{
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .match {{
                background: #d4edda;
                color: #155724;
            }}
            .mismatch {{
                background: #f8d7da;
                color: #721c24;
            }}
            .not-available {{
                background: #f8f9fa;
                color: #6c757d;
                text-align: center;
                padding: 40px;
                font-style: italic;
            }}
            .topic-section {{
                margin: 40px 0;
                padding: 20px 0;
                border-top: 3px solid #3498db;
            }}
            .topic-title {{
                background: #3498db;
                color: white;
                padding: 15px;
                margin: -20px 0 20px 0;
                border-radius: 5px;
                font-size: 1.3em;
                font-weight: bold;
            }}
            .case-count {{
                float: right;
                background: rgba(255,255,255,0.2);
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.8em;
            }}
            .filter-info {{
                background: #e3f2fd;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Vignette Analysis - Manual Evaluation</h1>
                <p>Generated on {timestamp}</p>
            </div>
            
            <div class="filter-info">
                <strong>Filtered Topics:</strong> {topics_display}
            </div>
            
            <div class="summary">
                <strong>Summary:</strong> {total_cases} cases across {topic_count} topics | 
                {both_models} cases with both models | {matches} matching decisions | {mismatches} mismatching decisions
            </div>
            
            {content}
        </div>
    </body>
    </html>
    """
    
    # Group cases by topic
    cases_by_topic = {}
    for case in cases:
        topic = case['topic']
        if topic not in cases_by_topic:
            cases_by_topic[topic] = []
        cases_by_topic[topic].append(case)
    
    # Generate content
    content = ""
    total_with_both = len(cases)  # All cases should have both models now
    total_matches = sum(1 for case in cases if case['decisions_match'])
    total_mismatches = sum(1 for case in cases if not case['decisions_match'])
    
    for topic in sorted(cases_by_topic.keys()):
        topic_cases = cases_by_topic[topic]
        content += f"""
        <div class="topic-section">
            <div class="topic-title">
                {topic}
                <span class="case-count">{len(topic_cases)} cases</span>
            </div>
        """
        
        for case in topic_cases:
            fields = case['fields']
            
            # Determine match status
            if case['decisions_match']:
                match_status = '<span class="match-indicator match">✓ MATCH</span>'
            else:
                match_status = '<span class="match-indicator mismatch">✗ MISMATCH</span>'
            
            content += f"""
            <div class="case-card">
                <div class="case-header">
                    <div class="case-title">{fields['name']}, {fields['age']}-year-old {fields['religion']} {fields['gender']} from {fields['country']}</div>
                    {match_status}
                </div>
                <div class="case-details">
                    <strong>Case Details:</strong> Age: {fields['age']} | Religion: {fields['religion']} | Gender: {fields['gender']} | Country: {fields['country']}
                </div>
                <div class="vignette-text">
                    {case['vignette_text']}
                </div>
                <div class="decisions-container">
            """
            
            # Pre-Brexit panel
            if case['pre_brexit']:
                pre_case = case['pre_brexit']
                content += f"""
                    <div class="decision-panel pre-brexit">
                        <h3>Pre-Brexit Model (2013-2016)</h3>
                        <div class="decision {pre_case['decision'].lower()}">{pre_case['decision'].upper()}</div>
                        <div class="reasoning">
                            <strong>Reasoning:</strong><br>
                            {pre_case['reasoning']}
                        </div>
                    </div>
                """
            else:
                content += """
                    <div class="decision-panel pre-brexit">
                        <h3>Pre-Brexit Model (2013-2016)</h3>
                        <div class="not-available">No data available for this case</div>
                    </div>
                """
            
            # Post-Brexit panel
            if case['post_brexit']:
                post_case = case['post_brexit']
                content += f"""
                    <div class="decision-panel post-brexit">
                        <h3>Post-Brexit Model (2019-2025)</h3>
                        <div class="decision {post_case['decision'].lower()}">{post_case['decision'].upper()}</div>
                        <div class="reasoning">
                            <strong>Reasoning:</strong><br>
                            {post_case['reasoning']}
                        </div>
                    </div>
                """
            else:
                content += """
                    <div class="decision-panel post-brexit">
                        <h3>Post-Brexit Model (2019-2025)</h3>
                        <div class="not-available">No data available for this case</div>
                    </div>
                """
            
            content += """
                </div>
            </div>
            """
        
        content += "</div>"
    
    # Fill in the template
    html_content = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        topics_display=", ".join(topics) if topics else "All topics",
        total_cases=len(cases),
        topic_count=len(cases_by_topic),
        both_models=total_with_both,
        matches=total_matches,
        mismatches=total_mismatches,
        content=content
    )
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")
    print(f"Open in browser to view: file://{Path(output_file).absolute()}")

def main():
    parser = argparse.ArgumentParser(description='Generate HTML viewer for vignette analysis')
    parser.add_argument('--topics', '-t', nargs='+', help='Topics to include (if not specified, includes all)')
    parser.add_argument('--output', '-o', default='vignette_viewer.html', help='Output HTML file (default: vignette_viewer.html)')
    parser.add_argument('--pre-brexit-file', default='../inference/results/processed/processed_subset_inference_llama3_8b_pre_brexit_2013_2016_instruct_20250623_120225_20250623_122325.json', help='Pre-Brexit processed results file')
    parser.add_argument('--post-brexit-file', default='../inference/results/processed/processed_subset_inference_llama3_8b_post_brexit_2019_2025_instruct_20250623_123821_20250623_124925.json', help='Post-Brexit processed results file')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading processed data...")
    pre_brexit_samples, pre_brexit_topic_lookup = load_processed_results(args.pre_brexit_file)
    post_brexit_samples, post_brexit_topic_lookup = load_processed_results(args.post_brexit_file)
    
    # Find matching samples
    matching_samples = find_matching_samples(pre_brexit_samples, post_brexit_samples, args.topics or [])
    
    print(f"Found {len(matching_samples)} cases")
    if args.topics:
        print(f"Filtered by topics: {', '.join(args.topics)}")
    
    # Generate HTML report
    generate_html_report(matching_samples, args.topics or [], args.output)

if __name__ == "__main__":
    main() 