import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

def load_processed_data(file_path: str) -> List[Dict]:
    """Load processed inference results."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data["processed_results"]

def create_case_lookup(data: List[Dict]) -> Dict[str, Dict]:
    """Create lookup dictionary with case key -> case data."""
    lookup = {}
    for case in data:
        metadata = case['metadata']
        key = f"{metadata['topic']}_{metadata['fields']['country']}_{metadata['fields']['age']}"
        lookup[key] = {
            'topic': metadata['topic'],
            'fields': metadata['fields'],
            'vignette_text': metadata['vignette_text'],
            'decision': case['decision'],
            'reasoning': case['reasoning'],
            'original_response': case['original_response']
        }
    return lookup

def filter_by_topics(pre_brexit_data: Dict, post_brexit_data: Dict, topics: List[str]) -> List[Dict]:
    """Filter cases by topics and combine both model results."""
    if not topics:
        # If no topics specified, include all
        topics = list(set([case['topic'] for case in pre_brexit_data.values()]))
    
    combined_cases = []
    
    # Get all unique case keys from both datasets
    all_keys = set(pre_brexit_data.keys()) | set(post_brexit_data.keys())
    
    for key in all_keys:
        pre_case = pre_brexit_data.get(key)
        post_case = post_brexit_data.get(key)
        
        # Get topic from whichever case exists
        case_topic = (pre_case or post_case)['topic']
        
        # Filter by topics
        if case_topic in topics:
            combined_case = {
                'case_key': key,
                'topic': case_topic,
                'fields': (pre_case or post_case)['fields'],
                'vignette_text': (pre_case or post_case)['vignette_text'],
                'pre_brexit': pre_case,
                'post_brexit': post_case,
                'has_both': pre_case is not None and post_case is not None,
                'decision_match': (pre_case and post_case and 
                                 pre_case['decision'] == post_case['decision'])
            }
            combined_cases.append(combined_case)
    
    # Sort by topic, then by country, then by age
    combined_cases.sort(key=lambda x: (x['topic'], x['fields']['country'], x['fields']['age']))
    return combined_cases

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
    total_with_both = sum(1 for case in cases if case['has_both'])
    total_matches = sum(1 for case in cases if case['decision_match'])
    total_mismatches = sum(1 for case in cases if case['has_both'] and not case['decision_match'])
    
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
            match_status = ""
            if case['has_both']:
                if case['decision_match']:
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
    pre_brexit_data = load_processed_data(args.pre_brexit_file)
    post_brexit_data = load_processed_data(args.post_brexit_file)
    
    # Create lookups
    pre_brexit_lookup = create_case_lookup(pre_brexit_data)
    post_brexit_lookup = create_case_lookup(post_brexit_data)
    
    # Filter by topics
    filtered_cases = filter_by_topics(pre_brexit_lookup, post_brexit_lookup, args.topics or [])
    
    print(f"Found {len(filtered_cases)} cases")
    if args.topics:
        print(f"Filtered by topics: {', '.join(args.topics)}")
    
    # Generate HTML report
    generate_html_report(filtered_cases, args.topics or [], args.output)

if __name__ == "__main__":
    main() 