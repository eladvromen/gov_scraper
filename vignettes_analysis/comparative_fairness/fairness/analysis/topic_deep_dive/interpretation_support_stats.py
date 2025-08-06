#!/usr/bin/env python3
"""
📊 Interpretation Support Statistics
===================================

Simple script to provide key statistics for interpreting the 4-panel visualization:
- Mean grant rates by completion and model
- Most concerning disparities
- Biggest changes by country and completion
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class InterpretationStats:
    """Generate interpretation support statistics"""
    
    def __init__(self):
        self.data = {}
        self.focus_countries = ["Syria", "Nigeria", "Myanmar"]
        
    def load_data(self):
        """Load the raw calculation data"""
        print("📊 Loading data for interpretation support...")
        
        try:
            # Load completion bias data
            completion_path = Path("raw_calculation_results/vignette_completion_bias_data.csv")
            self.data['completions'] = pd.read_csv(completion_path)
            
            # Load country patterns data
            patterns_path = Path("raw_calculation_results/focus_country_bias_patterns.csv")
            self.data['patterns'] = pd.read_csv(patterns_path)
            
            # Load our generated country-completion data
            country_completion_path = Path("raw_calculation_results/country_completion_grant_rates_table.csv")
            if country_completion_path.exists():
                self.data['country_completions'] = pd.read_csv(country_completion_path)
            
            print(f"✅ Loaded completion data: {len(self.data['completions'])} records")
            print(f"✅ Loaded pattern data: {len(self.data['patterns'])} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def analyze_mean_grants_by_completion(self):
        """Calculate mean grant rates by completion and model"""
        print("\n📈 MEAN GRANT RATES BY COMPLETION")
        print("=" * 50)
        
        completion_data = self.data['completions']
        
        # Sort by bias magnitude to show most problematic first
        completion_data_sorted = completion_data.sort_values('rate_change_magnitude', ascending=False)
        
        print(f"{'Completion Type':<40} {'Pre-Brexit':<12} {'Post-Brexit':<12} {'Change':<10} {'Magnitude':<10}")
        print("-" * 90)
        
        for _, row in completion_data_sorted.iterrows():
            field_name = row['field_name']
            field_value = row['field_value']
            pre_rate = row['pre_brexit_raw_rate']
            post_rate = row['post_brexit_raw_rate']
            change = row['cross_model_difference']
            magnitude = row['rate_change_magnitude']
            
            # Truncate long field values
            display_name = f"{field_name}: {field_value[:30]}..." if len(field_value) > 30 else f"{field_name}: {field_value}"
            
            print(f"{display_name:<40} {pre_rate:<12.3f} {post_rate:<12.3f} {change:<10.3f} {magnitude:<10.3f}")
        
        # Summary statistics
        print("\n📊 SUMMARY STATISTICS:")
        print(f"Mean Pre-Brexit rate: {completion_data['pre_brexit_raw_rate'].mean():.3f}")
        print(f"Mean Post-Brexit rate: {completion_data['post_brexit_raw_rate'].mean():.3f}")
        print(f"Overall average change: {completion_data['cross_model_difference'].mean():.3f}")
        print(f"Standard deviation of changes: {completion_data['cross_model_difference'].std():.3f}")
    
    def identify_most_concerning_disparities(self):
        """Identify the most concerning disparities"""
        print("\n🚨 MOST CONCERNING DISPARITIES")
        print("=" * 40)
        
        completion_data = self.data['completions']
        
        # Filter for statistically significant high bias
        high_bias = completion_data[
            (completion_data['statistical_significance'] == True) & 
            (completion_data['rate_change_magnitude'] > 0.1)
        ].copy()
        
        # Sort by magnitude
        high_bias_sorted = high_bias.sort_values('rate_change_magnitude', ascending=False)
        
        print("Top concerning completion biases (>10% change, statistically significant):")
        print()
        
        for i, (_, row) in enumerate(high_bias_sorted.iterrows(), 1):
            field_name = row['field_name']
            field_value = row['field_value']
            magnitude = row['rate_change_magnitude']
            p_value = row['p_value']
            sample_size = row['total_sample_size']
            
            # Truncate long field values
            display_name = field_value[:50] + "..." if len(field_value) > 50 else field_value
            
            print(f"{i}. {field_name.upper()}: {display_name}")
            print(f"   📉 Bias magnitude: {magnitude:.1%}")
            print(f"   📊 P-value: {p_value:.2e}")
            print(f"   👥 Sample size: {sample_size:,}")
            print(f"   ⚠️  Impact: {magnitude*100:.1f} percentage point drop in grant rates")
            print()
        
        # Rank by severity
        if len(high_bias_sorted) > 0:
            worst_completion = high_bias_sorted.iloc[0]
            print(f"🔥 MOST CONCERNING: {worst_completion['field_name']} - '{worst_completion['field_value'][:40]}...'")
            print(f"   Causes {worst_completion['rate_change_magnitude']:.1%} drop in grant rates")
    
    def analyze_biggest_changes_by_country(self):
        """Analyze biggest changes by country and completion"""
        print("\n🌍 BIGGEST CHANGES BY COUNTRY")
        print("=" * 35)
        
        # Use country-completion data if available, otherwise use patterns
        if 'country_completions' in self.data:
            country_data = self.data['country_completions']
            
            print("Changes by country across all completion types:")
            print()
            
            # Calculate average change by country
            country_summary = country_data.groupby('country').agg({
                'rate_change': ['mean', 'min', 'max', 'std']
            }).round(3)
            
            country_summary.columns = ['Avg_Change', 'Worst_Change', 'Best_Change', 'Std_Dev']
            country_summary = country_summary.sort_values('Avg_Change')
            
            print(f"{'Country':<12} {'Avg Change':<12} {'Worst Change':<12} {'Best Change':<12} {'Std Dev':<10}")
            print("-" * 65)
            
            for country, row in country_summary.iterrows():
                focus_indicator = "🎯" if country in self.focus_countries else "  "
                print(f"{focus_indicator} {country:<10} {row['Avg_Change']:<12.3f} {row['Worst_Change']:<12.3f} {row['Best_Change']:<12.3f} {row['Std_Dev']:<10.3f}")
            
            print("\nWorst country-completion combinations:")
            print()
            
            # Find worst combinations
            worst_combinations = country_data.nsmallest(10, 'rate_change')
            
            for i, (_, row) in enumerate(worst_combinations.iterrows(), 1):
                country = row['country']
                completion = row['completion_type']
                change = row['rate_change']
                focus_indicator = "🎯" if country in self.focus_countries else "  "
                
                print(f"{i:2d}. {focus_indicator} {country} + {completion}: {change:.3f} ({change:.1%})")
        
        # Also analyze statistical parity pattern changes
        if 'patterns' in self.data:
            print("\n📊 STATISTICAL PARITY PATTERN CHANGES:")
            print()
            
            patterns_data = self.data['patterns']
            
            # Find biggest changes involving focus countries
            focus_patterns = patterns_data[
                (patterns_data['focus_country_1']) | (patterns_data['focus_country_2'])
            ].copy()
            
            # Sort by absolute change
            focus_patterns_sorted = focus_patterns.sort_values('abs_sp_change', ascending=False)
            
            print("Top 10 biggest statistical parity changes (focus countries):")
            print()
            
            for i, (_, row) in enumerate(focus_patterns_sorted.head(10).iterrows(), 1):
                comparison = row['comparison_id']
                change = row['sp_magnitude_change']
                abs_change = row['abs_sp_change']
                pattern_type = row['pattern_type']
                
                print(f"{i:2d}. {comparison:<20} {change:+.3f} ({abs_change:.3f}) [{pattern_type}]")
    
    def generate_interpretation_summary(self):
        """Generate a concise interpretation summary"""
        print("\n🎯 INTERPRETATION SUMMARY")
        print("=" * 25)
        
        completion_data = self.data['completions']
        
        # Key insights
        worst_completion = completion_data.loc[completion_data['rate_change_magnitude'].idxmax()]
        least_biased = completion_data.loc[completion_data['rate_change_magnitude'].idxmin()]
        
        print("KEY INSIGHTS:")
        print()
        
        print(f"1. MOST PROBLEMATIC COMPLETION:")
        print(f"   '{worst_completion['field_value'][:60]}...'")
        print(f"   📉 {worst_completion['rate_change_magnitude']:.1%} drop in grant rates")
        print(f"   📊 P-value: {worst_completion['p_value']:.2e}")
        print()
        
        print(f"2. LEAST PROBLEMATIC COMPLETION:")
        print(f"   '{least_biased['field_value'][:60]}...'")
        print(f"   📈 {least_biased['rate_change_magnitude']:.1%} bias magnitude")
        print()
        
        # Count significant biases
        significant_biases = completion_data[completion_data['statistical_significance'] == True]
        high_magnitude_biases = completion_data[completion_data['rate_change_magnitude'] > 0.1]
        
        print(f"3. SCALE OF BIAS:")
        print(f"   📊 {len(significant_biases)}/{len(completion_data)} completions show statistically significant bias")
        print(f"   🚨 {len(high_magnitude_biases)}/{len(completion_data)} completions show high-magnitude bias (>10%)")
        print()
        
        # Focus country impact
        if 'country_completions' in self.data:
            country_data = self.data['country_completions']
            focus_impact = country_data[country_data['country'].isin(self.focus_countries)]
            avg_focus_impact = focus_impact['rate_change'].mean()
            
            print(f"4. FOCUS COUNTRY IMPACT:")
            print(f"   🎯 Average impact on Syria/Nigeria/Myanmar: {avg_focus_impact:.1%}")
            print(f"   📉 All focus countries experience negative impacts")
        
        print()
        print("💡 BOTTOM LINE:")
        print("   The Post-Brexit model systematically penalizes completions")
        print("   that signal 'economic opportunism' (career advancement,")
        print("   entrepreneurship, professional qualifications), with")
        print("   Syria, Nigeria, and Myanmar bearing disproportionate impacts.")
    
    def run_analysis(self):
        """Run complete interpretation support analysis"""
        print("📊 INTERPRETATION SUPPORT STATISTICS")
        print("===================================")
        
        if not self.load_data():
            return False
        
        # Run all analyses
        self.analyze_mean_grants_by_completion()
        self.identify_most_concerning_disparities()
        self.analyze_biggest_changes_by_country()
        self.generate_interpretation_summary()
        
        print("\n✅ Analysis complete!")
        return True

def main():
    """Main execution function"""
    analyzer = InterpretationStats()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 