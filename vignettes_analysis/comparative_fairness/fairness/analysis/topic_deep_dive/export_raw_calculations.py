#!/usr/bin/env python3
"""
📊 Export Raw Calculation Results
=================================

Export all raw calculation data from bias pattern analyses
to CSV and JSON formats for external analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class RawDataExporter:
    """Export raw calculation results from bias analyses"""
    
    def __init__(self, focus_countries: List[str], focus_topic: str):
        self.focus_countries = focus_countries
        self.focus_topic = focus_topic
        self.data = {}
        self.output_dir = Path("raw_calculation_results")
        
    def load_source_data(self):
        """Load all source data for raw extraction"""
        print("📊 Loading source data for raw extraction...")
        
        try:
            # Core fairness comparisons (FDR-corrected)
            fairness_path = Path("../../outputs/vector_drift/deduplicated_fairness_comparisons.csv")
            self.data['fairness_comparisons'] = pd.read_csv(fairness_path)
            print(f"✅ Loaded {len(self.data['fairness_comparisons'])} fairness comparisons")
            
            # Grant rate analysis by field
            grant_rate_path = Path("/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/normative/outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced_FIXED.csv")
            self.data['grant_rates'] = pd.read_csv(grant_rate_path)
            print(f"✅ Loaded {len(self.data['grant_rates'])} grant rate field analyses")
            
            # Unified fairness dataframe
            unified_path = Path("../../outputs/unified/unified_fairness_dataframe_topic_granular.csv")
            self.data['unified_fairness'] = pd.read_csv(unified_path)
            print(f"✅ Loaded {len(self.data['unified_fairness'])} unified fairness records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading source data: {e}")
            return False
    
    def extract_focus_country_patterns(self) -> pd.DataFrame:
        """Extract raw pattern data for focus countries"""
        print("🎯 Extracting focus country pattern data...")
        
        fairness_data = self.data['fairness_comparisons']
        
        # Filter for focus topic (using flexible search since topic names are truncated)
        topic_search = "Intentions regarding work"  # Simplified search term
        topic_mask = fairness_data['comparison_label'].str.contains(topic_search, case=False, na=False)
        topic_data = fairness_data[topic_mask].copy()
        
        focus_patterns = []
        
        for _, row in topic_data.iterrows():
            label = row['comparison_label']
            
            # Parse comparison label
            if '(' in label and ')' in label and '[' in label:
                main_part = label.split(' (')[0]
                remainder = label.split(' (')[1]
                attribute = remainder.split(')')[0]
                topic = remainder.split('[')[1].split(']')[0]
                
                if '_vs_' in main_part and attribute == 'country':
                    group1, group2 = main_part.split('_vs_')
                    
                    # Check if either group is in focus countries
                    if group1 in self.focus_countries or group2 in self.focus_countries:
                        
                        # Determine pattern classification
                        if row['sp_gained_significance']:
                            pattern_type = "Newly_Emerged"
                        elif row['sp_lost_significance']:
                            pattern_type = "Disappeared"
                        elif row['sp_both_significant']:
                            pattern_type = "Persistent"
                        else:
                            pattern_type = "Non_Significant"
                        
                        # Determine bias direction
                        if row['sp_magnitude_difference'] > 0:
                            bias_direction = f"Favors_{group2}_over_{group1}"
                        elif row['sp_magnitude_difference'] < 0:
                            bias_direction = f"Favors_{group1}_over_{group2}"
                        else:
                            bias_direction = "No_Bias"
                        
                        focus_patterns.append({
                            'comparison_id': f"{group1}_vs_{group2}",
                            'group_1': group1,
                            'group_2': group2,
                            'topic': topic,
                            'attribute': attribute,
                            'pre_brexit_sp_magnitude': row['pre_brexit_sp_magnitude'],
                            'post_brexit_sp_magnitude': row['post_brexit_sp_magnitude'],
                            'sp_magnitude_change': row['sp_magnitude_difference'],
                            'abs_sp_change': abs(row['sp_magnitude_difference']),
                            'pre_brexit_significant': row['pre_brexit_sp_significance'],
                            'post_brexit_significant': row['post_brexit_sp_significance'],
                            'gained_significance': row['sp_gained_significance'],
                            'lost_significance': row['sp_lost_significance'],
                            'both_significant': row['sp_both_significant'],
                            'neither_significant': row['sp_neither_significant'],
                            'pattern_type': pattern_type,
                            'bias_direction': bias_direction,
                            'is_high_magnitude': abs(row['sp_magnitude_difference']) > 0.15,
                            'focus_country_1': group1 in self.focus_countries,
                            'focus_country_2': group2 in self.focus_countries,
                            'analysis_timestamp': datetime.now().isoformat()
                        })
        
        patterns_df = pd.DataFrame(focus_patterns)
        print(f"📊 Extracted {len(patterns_df)} focus country patterns")
        
        return patterns_df
    
    def extract_vignette_completion_data(self) -> pd.DataFrame:
        """Extract raw vignette completion bias data"""
        print("📝 Extracting vignette completion data...")
        
        grant_data = self.data['grant_rates']
        
        # Filter for work-related topic
        work_data = grant_data[
            grant_data['topic'].str.contains("Intentions regarding work", case=False, na=False)
        ].copy()
        
        completion_data = []
        
        for _, row in work_data.iterrows():
            completion_data.append({
                'field_name': row['field_name'],
                'field_type': row['field_type'],
                'field_value': row['field_value'],
                'topic': row['topic'],
                'pre_brexit_raw_rate': row['pre_brexit_raw_rate'],
                'post_brexit_raw_rate': row['post_brexit_raw_rate'],
                'pre_brexit_normalized': row['pre_brexit_normalized'],
                'post_brexit_normalized': row['post_brexit_normalized'],
                'cross_model_difference': row['cross_model_difference'],
                'pre_brexit_sample_size': row['pre_brexit_sample_size'],
                'post_brexit_sample_size': row['post_brexit_sample_size'],
                'total_sample_size': row['pre_brexit_sample_size'] + row['post_brexit_sample_size'],
                'statistical_significance': row['statistical_significance'],
                'p_value': row['p_value'],
                'favors_model': row['favors_model'],
                'rate_change_magnitude': abs(row['cross_model_difference']),
                'is_high_bias': row['statistical_significance'] and abs(row['cross_model_difference']) > 0.1,
                'bias_direction': 'Post_Brexit_Higher' if row['cross_model_difference'] > 0 else 'Pre_Brexit_Higher',
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        completion_df = pd.DataFrame(completion_data)
        print(f"📊 Extracted {len(completion_df)} completion bias records")
        
        return completion_df
    
    def extract_statistical_summary(self, patterns_df: pd.DataFrame, 
                                  completion_df: pd.DataFrame) -> Dict:
        """Extract statistical summary of findings"""
        print("📈 Computing statistical summary...")
        
        summary = {
            'analysis_metadata': {
                'focus_countries': self.focus_countries,
                'focus_topic': self.focus_topic,
                'analysis_date': datetime.now().isoformat(),
                'total_patterns_analyzed': len(patterns_df),
                'total_completions_analyzed': len(completion_df)
            },
            'pattern_statistics': {},
            'completion_statistics': {},
            'country_specific_summary': {}
        }
        
        # Pattern statistics (handle empty dataframes)
        if len(patterns_df) > 0 and 'pattern_type' in patterns_df.columns:
            summary['pattern_statistics'] = {
                'newly_emerged_patterns': int(patterns_df['pattern_type'].eq('Newly_Emerged').sum()),
                'disappeared_patterns': int(patterns_df['pattern_type'].eq('Disappeared').sum()),
                'persistent_patterns': int(patterns_df['pattern_type'].eq('Persistent').sum()),
                'non_significant_patterns': int(patterns_df['pattern_type'].eq('Non_Significant').sum()),
                'high_magnitude_patterns': int(patterns_df['is_high_magnitude'].sum()),
                'mean_sp_change': float(patterns_df['sp_magnitude_change'].mean()),
                'median_sp_change': float(patterns_df['sp_magnitude_change'].median()),
                'max_sp_change': float(patterns_df['abs_sp_change'].max()),
                'std_sp_change': float(patterns_df['sp_magnitude_change'].std())
            }
        else:
            summary['pattern_statistics'] = {
                'newly_emerged_patterns': 0,
                'disappeared_patterns': 0,
                'persistent_patterns': 0,
                'non_significant_patterns': 0,
                'high_magnitude_patterns': 0,
                'mean_sp_change': 0.0,
                'median_sp_change': 0.0,
                'max_sp_change': 0.0,
                'std_sp_change': 0.0
            }
        
        # Completion statistics (handle empty dataframes)
        if len(completion_df) > 0:
            summary['completion_statistics'] = {
                'high_bias_completions': int(completion_df['is_high_bias'].sum()),
                'work_intentions_completions': int(completion_df['field_name'].eq('work intentions').sum()),
                'profession_completions': int(completion_df['field_name'].eq('profession').sum()),
                'mean_rate_change': float(completion_df['cross_model_difference'].mean()),
                'median_rate_change': float(completion_df['cross_model_difference'].median()),
                'max_rate_change': float(completion_df['rate_change_magnitude'].max()),
                'std_rate_change': float(completion_df['cross_model_difference'].std())
            }
        else:
            summary['completion_statistics'] = {
                'high_bias_completions': 0,
                'work_intentions_completions': 0,
                'profession_completions': 0,
                'mean_rate_change': 0.0,
                'median_rate_change': 0.0,
                'max_rate_change': 0.0,
                'std_rate_change': 0.0
            }
        
        # Country-specific statistics
        for country in self.focus_countries:
            country_patterns = patterns_df[
                (patterns_df['group_1'] == country) | (patterns_df['group_2'] == country)
            ]
            
            gained_patterns = country_patterns[
                ((country_patterns['group_1'] == country) & (country_patterns['sp_magnitude_change'] < 0)) |
                ((country_patterns['group_2'] == country) & (country_patterns['sp_magnitude_change'] > 0))
            ]
            
            lost_patterns = country_patterns[
                ((country_patterns['group_1'] == country) & (country_patterns['sp_magnitude_change'] > 0)) |
                ((country_patterns['group_2'] == country) & (country_patterns['sp_magnitude_change'] < 0))
            ]
            
            summary['country_specific_summary'][country] = {
                'total_patterns': int(len(country_patterns)),
                'advantageous_changes': int(len(gained_patterns)),
                'disadvantageous_changes': int(len(lost_patterns)),
                'net_change': int(len(gained_patterns) - len(lost_patterns)),
                'mean_sp_impact': float(country_patterns['sp_magnitude_change'].mean()) if len(country_patterns) > 0 else 0,
                'high_magnitude_patterns': int(country_patterns['is_high_magnitude'].sum())
            }
        
        return summary
    
    def export_detailed_calculations(self, patterns_df: pd.DataFrame, 
                                   completion_df: pd.DataFrame, summary: Dict):
        """Export all detailed calculations"""
        print("💾 Exporting detailed calculations...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export focus country patterns (main results)
        patterns_df.to_csv(self.output_dir / 'focus_country_bias_patterns.csv', index=False)
        patterns_df.to_json(self.output_dir / 'focus_country_bias_patterns.json', orient='records', indent=2)
        
        # 2. Export vignette completion data
        completion_df.to_csv(self.output_dir / 'vignette_completion_bias_data.csv', index=False)
        completion_df.to_json(self.output_dir / 'vignette_completion_bias_data.json', orient='records', indent=2)
        
        # 3. Export statistical summary
        with open(self.output_dir / 'statistical_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 4. Export pivot tables for easy analysis
        
        # Pattern type by country pivot
        pattern_pivot = patterns_df.pivot_table(
            values='abs_sp_change', 
            index=['group_1', 'group_2'], 
            columns='pattern_type', 
            aggfunc='mean', 
            fill_value=0
        )
        pattern_pivot.to_csv(self.output_dir / 'pattern_type_by_country_pivot.csv')
        
        # Completion bias by field pivot  
        completion_pivot = completion_df.pivot_table(
            values='cross_model_difference',
            index='field_value',
            columns='field_name',
            aggfunc='first',
            fill_value=0
        )
        completion_pivot.to_csv(self.output_dir / 'completion_bias_by_field_pivot.csv')
        
        # 5. Export high-priority cases for manual review
        high_priority = patterns_df[patterns_df['is_high_magnitude'] == True].copy()
        high_priority = high_priority.sort_values('abs_sp_change', ascending=False)
        high_priority.to_csv(self.output_dir / 'high_priority_cases_for_manual_review.csv', index=False)
        
        # 6. Export completion-level high bias cases
        high_bias_completions = completion_df[completion_df['is_high_bias'] == True].copy()
        high_bias_completions = high_bias_completions.sort_values('rate_change_magnitude', ascending=False)
        high_bias_completions.to_csv(self.output_dir / 'high_bias_completions.csv', index=False)
        
        print(f"✅ Exported {len(patterns_df)} pattern records")
        print(f"✅ Exported {len(completion_df)} completion records") 
        print(f"✅ Exported statistical summary")
        print(f"✅ Exported {len(high_priority)} high-priority cases")
        print(f"✅ Exported {len(high_bias_completions)} high-bias completions")
        
    def create_data_dictionary(self):
        """Create data dictionary explaining all exported fields"""
        print("📚 Creating data dictionary...")
        
        dictionary = {
            "data_dictionary": {
                "focus_country_bias_patterns.csv": {
                    "description": "Raw statistical parity patterns for focus countries",
                    "fields": {
                        "comparison_id": "Unique identifier for country comparison (e.g., 'Syria_vs_Myanmar')",
                        "group_1": "First country in comparison",
                        "group_2": "Second country in comparison", 
                        "pre_brexit_sp_magnitude": "Statistical parity value in pre-Brexit model (-1 to 1)",
                        "post_brexit_sp_magnitude": "Statistical parity value in post-Brexit model (-1 to 1)",
                        "sp_magnitude_change": "Change in statistical parity (post - pre)",
                        "abs_sp_change": "Absolute magnitude of change",
                        "pre_brexit_significant": "Whether pre-Brexit bias was FDR-significant",
                        "post_brexit_significant": "Whether post-Brexit bias was FDR-significant",
                        "pattern_type": "Classification: Newly_Emerged, Disappeared, Persistent, Non_Significant",
                        "bias_direction": "Which country/group is favored",
                        "is_high_magnitude": "Whether |change| > 0.15 (threshold for major bias shift)"
                    }
                },
                "vignette_completion_bias_data.csv": {
                    "description": "Raw bias data for specific vignette field completions",
                    "fields": {
                        "field_name": "Vignette field (e.g., 'work intentions', 'profession')",
                        "field_value": "Specific completion text",
                        "pre_brexit_raw_rate": "Grant rate in pre-Brexit model (0-1)",
                        "post_brexit_raw_rate": "Grant rate in post-Brexit model (0-1)", 
                        "cross_model_difference": "Difference in grant rates (post - pre)",
                        "statistical_significance": "Whether difference is statistically significant",
                        "p_value": "Statistical significance p-value",
                        "is_high_bias": "Whether |difference| > 0.1 AND statistically significant"
                    }
                },
                "statistical_summary.json": {
                    "description": "Aggregate statistics and summary metrics",
                    "sections": {
                        "pattern_statistics": "Counts and statistics for bias patterns",
                        "completion_statistics": "Statistics for vignette completion bias",
                        "country_specific_summary": "Per-country impact analysis"
                    }
                },
                "high_priority_cases_for_manual_review.csv": {
                    "description": "Cases with highest bias magnitude requiring manual investigation",
                    "note": "Sorted by absolute statistical parity change, filtered for |change| > 0.15"
                },
                "high_bias_completions.csv": {
                    "description": "Vignette completions with statistically significant bias > 10%",
                    "note": "These are the specific field values driving bias patterns"
                }
            },
            "methodology": {
                "statistical_parity": "Difference in positive decision rates between groups, range [-1, 1]",
                "fdr_correction": "False Discovery Rate correction applied to significance testing",
                "bias_threshold": "High bias defined as |change| > 0.10 for grant rates, > 0.15 for SP",
                "focus_countries": self.focus_countries,
                "focus_topic": self.focus_topic
            }
        }
        
        with open(self.output_dir / 'data_dictionary.json', 'w') as f:
            json.dump(dictionary, f, indent=2)
    
    def run_export(self) -> bool:
        """Run complete raw data export"""
        print(f"📊 Starting Raw Data Export")
        print(f"Focus: {', '.join(self.focus_countries)} in '{self.focus_topic}'")
        print("=" * 60)
        
        # Load source data
        if not self.load_source_data():
            return False
        
        # Extract raw calculations
        patterns_df = self.extract_focus_country_patterns()
        completion_df = self.extract_vignette_completion_data()
        summary = self.extract_statistical_summary(patterns_df, completion_df)
        
        # Export everything
        self.export_detailed_calculations(patterns_df, completion_df, summary)
        self.create_data_dictionary()
        
        print(f"\n✅ Raw Data Export Complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print("📊 Files generated:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   • {file.name}")
        print("=" * 60)
        
        return True

def main():
    """Main execution function"""
    
    # Configuration
    focus_countries = ["Syria", "Nigeria", "Myanmar"]
    focus_topic = "Intentions regarding work in the UK"
    
    # Initialize exporter
    exporter = RawDataExporter(focus_countries, focus_topic)
    
    # Run export
    success = exporter.run_export()
    
    if success:
        print("\n🎯 Raw Calculation Results Available:")
        print("• focus_country_bias_patterns.csv - Main statistical parity results")
        print("• vignette_completion_bias_data.csv - Completion-level bias data")
        print("• statistical_summary.json - Aggregate statistics")
        print("• high_priority_cases_for_manual_review.csv - Top cases to investigate")
        print("• high_bias_completions.csv - Problematic vignette completions")
        print("• data_dictionary.json - Explains all fields and methodology")
    else:
        print("\n❌ Export failed. Check data paths and requirements.")

if __name__ == "__main__":
    main() 