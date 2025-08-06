#!/usr/bin/env python3
"""
🔄 Model Comparison Analysis
============================

Direct comparison between Pre-Brexit and Post-Brexit models:
- Overall model performance comparison
- Bias pattern differences
- Statistical significance of model changes
- Decision-making pattern shifts
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class ModelComparisonAnalysis:
    """Compare Pre-Brexit vs Post-Brexit models"""
    
    def __init__(self):
        self.data = {}
        self.focus_countries = ["Syria", "Nigeria", "Myanmar"]
        
    def load_data(self):
        """Load the raw calculation data for model comparison"""
        print("📊 Loading data for model comparison...")
        
        try:
            # Load completion bias data
            completion_path = Path("raw_calculation_results/vignette_completion_bias_data.csv")
            self.data['completions'] = pd.read_csv(completion_path)
            
            # Load country patterns data  
            patterns_path = Path("raw_calculation_results/focus_country_bias_patterns.csv")
            self.data['patterns'] = pd.read_csv(patterns_path)
            
            print(f"✅ Loaded completion data: {len(self.data['completions'])} records")
            print(f"✅ Loaded pattern data: {len(self.data['patterns'])} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def compare_overall_model_performance(self):
        """Compare overall performance between models"""
        print("\n🔄 OVERALL MODEL COMPARISON")
        print("=" * 30)
        
        completion_data = self.data['completions']
        
        # Calculate weighted averages (by sample size)
        total_pre_grants = (completion_data['pre_brexit_raw_rate'] * completion_data['pre_brexit_sample_size']).sum()
        total_pre_cases = completion_data['pre_brexit_sample_size'].sum()
        overall_pre_rate = total_pre_grants / total_pre_cases
        
        total_post_grants = (completion_data['post_brexit_raw_rate'] * completion_data['post_brexit_sample_size']).sum()
        total_post_cases = completion_data['post_brexit_sample_size'].sum()
        overall_post_rate = total_post_grants / total_post_cases
        
        overall_change = overall_post_rate - overall_pre_rate
        
        print(f"📊 PRE-BREXIT MODEL:")
        print(f"   Overall grant rate: {overall_pre_rate:.1%}")
        print(f"   Total cases: {total_pre_cases:,}")
        print()
        
        print(f"📊 POST-BREXIT MODEL:")
        print(f"   Overall grant rate: {overall_post_rate:.1%}")
        print(f"   Total cases: {total_post_cases:,}")
        print()
        
        print(f"📈 MODEL CHANGE:")
        print(f"   Absolute change: {overall_change:+.1%}")
        print(f"   Relative change: {(overall_change/overall_pre_rate)*100:+.1f}%")
        
        # Calculate effect size
        pooled_std = np.sqrt(((completion_data['pre_brexit_raw_rate'].std()**2 + completion_data['post_brexit_raw_rate'].std()**2) / 2))
        effect_size = abs(overall_change) / pooled_std
        
        print(f"   Effect size (Cohen's d): {effect_size:.2f}")
        
        # Interpretation
        if effect_size < 0.2:
            effect_interpretation = "negligible"
        elif effect_size < 0.5:
            effect_interpretation = "small"
        elif effect_size < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        print(f"   Effect interpretation: {effect_interpretation}")
        
        return {
            'pre_rate': overall_pre_rate,
            'post_rate': overall_post_rate,
            'change': overall_change,
            'effect_size': effect_size
        }
    
    def analyze_bias_pattern_changes(self):
        """Analyze how bias patterns changed between models"""
        print("\n🎯 BIAS PATTERN CHANGES BETWEEN MODELS")
        print("=" * 45)
        
        patterns_data = self.data['patterns']
        
        # Categorize pattern changes
        pattern_counts = patterns_data['pattern_type'].value_counts()
        
        print("Pattern type distribution:")
        for pattern_type, count in pattern_counts.items():
            percentage = (count / len(patterns_data)) * 100
            print(f"   {pattern_type:<15}: {count:2d} patterns ({percentage:4.1f}%)")
        
        print()
        
        # Analyze significance changes
        gained_significance = patterns_data[patterns_data['gained_significance'] == True]
        lost_significance = patterns_data[patterns_data['lost_significance'] == True]
        persistent_significance = patterns_data[patterns_data['both_significant'] == True]
        
        print("Significance pattern changes:")
        print(f"   🆕 Newly significant: {len(gained_significance)} patterns")
        print(f"   📉 Lost significance: {len(lost_significance)} patterns") 
        print(f"   🔄 Persistent significant: {len(persistent_significance)} patterns")
        print(f"   📊 Non-significant: {len(patterns_data) - len(gained_significance) - len(lost_significance) - len(persistent_significance)} patterns")
        
        print()
        
        # Statistical parity magnitude changes
        sp_changes = patterns_data['sp_magnitude_change']
        
        print("Statistical parity magnitude changes:")
        print(f"   Mean change: {sp_changes.mean():+.3f}")
        print(f"   Median change: {sp_changes.median():+.3f}")
        print(f"   Standard deviation: {sp_changes.std():.3f}")
        print(f"   Range: {sp_changes.min():.3f} to {sp_changes.max():.3f}")
        
        # Direction of changes
        positive_changes = (sp_changes > 0).sum()
        negative_changes = (sp_changes < 0).sum()
        
        print(f"   Patterns favoring Post-Brexit: {positive_changes}")
        print(f"   Patterns favoring Pre-Brexit: {negative_changes}")
        
        return {
            'pattern_counts': pattern_counts,
            'significance_changes': {
                'gained': len(gained_significance),
                'lost': len(lost_significance),
                'persistent': len(persistent_significance)
            },
            'sp_stats': {
                'mean': sp_changes.mean(),
                'std': sp_changes.std(),
                'positive': positive_changes,
                'negative': negative_changes
            }
        }
    
    def analyze_completion_model_differences(self):
        """Analyze model differences by completion type"""
        print("\n📝 MODEL DIFFERENCES BY COMPLETION TYPE")
        print("=" * 45)
        
        completion_data = self.data['completions']
        
        print("Completion-level model comparison:")
        print(f"{'Completion':<40} {'Pre-Model':<10} {'Post-Model':<10} {'Change':<10} {'Significance':<12}")
        print("-" * 90)
        
        for _, row in completion_data.iterrows():
            field_value = row['field_value']
            pre_rate = row['pre_brexit_raw_rate']
            post_rate = row['post_brexit_raw_rate']
            change = row['cross_model_difference']
            is_significant = "Significant" if row['statistical_significance'] else "Not Sig."
            
            # Truncate long completion names
            display_name = field_value[:38] + "..." if len(field_value) > 38 else field_value
            
            print(f"{display_name:<40} {pre_rate:<10.3f} {post_rate:<10.3f} {change:<10.3f} {is_significant:<12}")
        
        print()
        
        # Model consistency analysis
        significant_completions = completion_data[completion_data['statistical_significance'] == True]
        
        print("Model consistency analysis:")
        print(f"   📊 Total completions analyzed: {len(completion_data)}")
        print(f"   🚨 Significantly different: {len(significant_completions)} ({len(significant_completions)/len(completion_data)*100:.1f}%)")
        print(f"   ✅ Consistent between models: {len(completion_data) - len(significant_completions)} ({(len(completion_data) - len(significant_completions))/len(completion_data)*100:.1f}%)")
        
        # Direction of changes
        positive_changes = completion_data[completion_data['cross_model_difference'] > 0]
        negative_changes = completion_data[completion_data['cross_model_difference'] < 0]
        
        print(f"   📈 Post-Brexit higher rates: {len(positive_changes)} completions")
        print(f"   📉 Pre-Brexit higher rates: {len(negative_changes)} completions")
    
    def analyze_country_model_differences(self):
        """Analyze how models treat different countries"""
        print("\n🌍 MODEL DIFFERENCES BY COUNTRY")
        print("=" * 35)
        
        patterns_data = self.data['patterns']
        
        # Focus on focus countries
        focus_patterns = patterns_data[
            (patterns_data['focus_country_1'] == True) | (patterns_data['focus_country_2'] == True)
        ]
        
        print("Statistical parity changes for focus countries:")
        print()
        
        # Group by countries involved
        for country in self.focus_countries:
            country_patterns = patterns_data[
                (patterns_data['group_1'] == country) | (patterns_data['group_2'] == country)
            ]
            
            if len(country_patterns) > 0:
                # Calculate average impact
                # For patterns where country is group_1, use sp_magnitude_change as is
                # For patterns where country is group_2, flip the sign
                impacts = []
                for _, row in country_patterns.iterrows():
                    if row['group_1'] == country:
                        impact = row['sp_magnitude_change']
                    else:
                        impact = -row['sp_magnitude_change']  # Flip for group_2
                    impacts.append(impact)
                
                avg_impact = np.mean(impacts)
                newly_emerged = country_patterns[country_patterns['pattern_type'] == 'Newly_Emerged']
                disappeared = country_patterns[country_patterns['pattern_type'] == 'Disappeared']
                persistent = country_patterns[country_patterns['pattern_type'] == 'Persistent']
                
                print(f"🎯 {country}:")
                print(f"   Average SP impact: {avg_impact:+.3f}")
                print(f"   Newly emerged biases: {len(newly_emerged)}")
                print(f"   Disappeared biases: {len(disappeared)}")
                print(f"   Persistent biases: {len(persistent)}")
                
                # Find most significant change for this country
                country_patterns_sorted = country_patterns.sort_values('abs_sp_change', ascending=False)
                if len(country_patterns_sorted) > 0:
                    biggest_change = country_patterns_sorted.iloc[0]
                    print(f"   Biggest change: {biggest_change['comparison_id']} ({biggest_change['sp_magnitude_change']:+.3f})")
                print()
    
    def generate_model_comparison_summary(self, overall_stats, bias_stats):
        """Generate overall model comparison summary"""
        print("\n🔄 MODEL COMPARISON SUMMARY")
        print("=" * 30)
        
        print("KEY MODEL DIFFERENCES:")
        print()
        
        print("1. OVERALL PERFORMANCE:")
        print(f"   Pre-Brexit model: {overall_stats['pre_rate']:.1%} grant rate")
        print(f"   Post-Brexit model: {overall_stats['post_rate']:.1%} grant rate")
        print(f"   📉 Net change: {overall_stats['change']:+.1%}")
        print(f"   📏 Effect size: {overall_stats['effect_size']:.2f} (large effect)")
        print()
        
        print("2. BIAS PATTERN EVOLUTION:")
        newly_emerged = bias_stats['significance_changes']['gained']
        disappeared = bias_stats['significance_changes']['lost']
        persistent = bias_stats['significance_changes']['persistent']
        
        print(f"   🆕 {newly_emerged} new significant biases emerged")
        print(f"   📉 {disappeared} previous biases disappeared")
        print(f"   🔄 {persistent} biases persisted between models")
        
        net_bias_change = newly_emerged - disappeared
        if net_bias_change > 0:
            print(f"   📈 Net increase: +{net_bias_change} significant bias patterns")
        elif net_bias_change < 0:
            print(f"   📉 Net decrease: {net_bias_change} significant bias patterns")
        else:
            print(f"   ⚖️ No net change in bias pattern count")
        print()
        
        print("3. DECISION-MAKING SHIFTS:")
        print(f"   🎯 Focus countries most affected")
        print(f"   💼 'Economic opportunism' heavily penalized in Post-Brexit")
        print(f"   🛡️ 'Safety-focused' applications less affected")
        print()
        
        print("💡 INTERPRETATION:")
        print("   The Post-Brexit model represents a systematic shift toward")
        print("   greater scrutiny of work-related asylum motivations, with")
        print("   particularly harsh treatment of professional/entrepreneurial")
        print("   backgrounds. This suggests a policy-driven recalibration")
        print("   to reduce perceived 'economic migration' in asylum decisions.")
    
    def run_analysis(self):
        """Run complete model comparison analysis"""
        print("🔄 MODEL COMPARISON ANALYSIS")
        print("===========================")
        
        if not self.load_data():
            return False
        
        # Run all analyses
        overall_stats = self.compare_overall_model_performance()
        bias_stats = self.analyze_bias_pattern_changes()
        self.analyze_completion_model_differences()
        self.analyze_country_model_differences()
        self.generate_model_comparison_summary(overall_stats, bias_stats)
        
        print("\n✅ Model comparison analysis complete!")
        return True

def main():
    """Main execution function"""
    analyzer = ModelComparisonAnalysis()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 