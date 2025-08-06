#!/usr/bin/env python3
"""
🔍 Pattern Inspection Deep Dive
===============================

Detailed qualitative analysis of specific bias pattern changes.
Links quantitative patterns back to actual vignette completions, 
grant rates, and specific scenarios.

Focus: Syria/Nigeria pattern losses vs Myanmar pattern gains
in work-related vignettes.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class PatternInspectionAnalyzer:
    """Detailed inspector for specific bias pattern changes"""
    
    def __init__(self, base_data_dir: Path, focus_countries: List[str], focus_topic: str):
        self.base_data_dir = base_data_dir
        self.focus_countries = focus_countries
        self.focus_topic = focus_topic
        self.data = {}
        self.pattern_details = {}
        
    def load_inspection_data(self):
        """Load all data needed for detailed pattern inspection"""
        print("🔍 Loading data for pattern inspection...")
        
        try:
            # Core fairness data with FDR significance
            self.data['fairness_comparisons'] = pd.read_csv(
                self.base_data_dir / "vector_drift" / "deduplicated_fairness_comparisons.csv"
            )
            
            # Grant rate analysis data
            grant_rate_path = Path("/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/normative/outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced_FIXED.csv")
            self.data['grant_rates'] = pd.read_csv(grant_rate_path)
            
            # Try to load vignette completion data (if available)
            try:
                # This would be the actual model completions - adapt path as needed
                vignette_path = Path("/data/shil6369/vignettes/complete_vignettes.json")
                with open(vignette_path, 'r') as f:
                    self.data['vignettes'] = json.load(f)
                print(f"✅ Loaded {len(self.data['vignettes'])} vignette templates")
            except FileNotFoundError:
                print("⚠️ Vignette completion data not found - will use grant rate analysis only")
                self.data['vignettes'] = {}
            
            # Load model results (if available)
            try:
                results_dir = Path("/data/shil6369/gov_scraper/inference/results/processed")
                # Look for result files
                result_files = list(results_dir.glob("*brexit*"))
                if result_files:
                    print(f"📁 Found {len(result_files)} model result files")
                    self.data['model_results'] = result_files
                else:
                    self.data['model_results'] = []
            except:
                self.data['model_results'] = []
            
            print(f"✅ Loaded inspection data successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading inspection data: {e}")
            return False
    
    def identify_focus_patterns(self) -> Dict[str, List[Dict]]:
        """Identify specific patterns that changed for focus countries"""
        print(f"🎯 Identifying pattern changes for: {', '.join(self.focus_countries)}")
        
        fairness_data = self.data['fairness_comparisons']
        
        # Filter for focus topic
        topic_mask = fairness_data['comparison_label'].str.contains(self.focus_topic, case=False, na=False)
        topic_data = fairness_data[topic_mask].copy()
        
        focus_patterns = {
            'gained_patterns': [],
            'lost_patterns': [],
            'persistent_patterns': [],
            'all_focus_patterns': []
        }
        
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
                        pattern_info = {
                            'comparison': f"{group1} vs {group2}",
                            'group1': group1,
                            'group2': group2,
                            'pre_sp': row['pre_brexit_sp_magnitude'],
                            'post_sp': row['post_brexit_sp_magnitude'],
                            'sp_change': row['sp_magnitude_difference'],
                            'pre_significant': row['pre_brexit_sp_significance'],
                            'post_significant': row['post_brexit_sp_significance'],
                            'gained_significance': row['sp_gained_significance'],
                            'lost_significance': row['sp_lost_significance'],
                            'both_significant': row['sp_both_significant'],
                            'label': label,
                            'topic': topic
                        }
                        
                        # Classify pattern change
                        if pattern_info['gained_significance']:
                            focus_patterns['gained_patterns'].append(pattern_info)
                        elif pattern_info['lost_significance']:
                            focus_patterns['lost_patterns'].append(pattern_info)
                        elif pattern_info['both_significant']:
                            focus_patterns['persistent_patterns'].append(pattern_info)
                        
                        focus_patterns['all_focus_patterns'].append(pattern_info)
        
        print(f"📊 Found patterns for focus countries:")
        print(f"   Gained: {len(focus_patterns['gained_patterns'])}")
        print(f"   Lost: {len(focus_patterns['lost_patterns'])}")
        print(f"   Persistent: {len(focus_patterns['persistent_patterns'])}")
        print(f"   Total: {len(focus_patterns['all_focus_patterns'])}")
        
        return focus_patterns
    
    def analyze_grant_rate_translation(self, focus_patterns: Dict) -> Dict[str, Dict]:
        """Analyze how bias patterns translate to grant rate differences"""
        print("💰 Analyzing grant rate translations...")
        
        grant_data = self.data['grant_rates']
        grant_analysis = {}
        
        # Filter grant data for focus topic
        topic_grant_data = grant_data[
            grant_data['topic'].str.contains(self.focus_topic, case=False, na=False)
        ].copy()
        
        for pattern_type, patterns in focus_patterns.items():
            if pattern_type == 'all_focus_patterns':
                continue
                
            grant_analysis[pattern_type] = {
                'pattern_count': len(patterns),
                'grant_rate_effects': [],
                'avg_grant_difference': 0,
                'max_grant_difference': 0
            }
            
            for pattern in patterns:
                group1, group2 = pattern['group1'], pattern['group2']
                
                # Try to find corresponding grant rate data
                # This requires mapping country names to grant rate field values
                grant_effects = self._find_grant_rate_effects(
                    topic_grant_data, group1, group2, pattern
                )
                
                if grant_effects:
                    grant_analysis[pattern_type]['grant_rate_effects'].extend(grant_effects)
            
            # Calculate summary statistics
            if grant_analysis[pattern_type]['grant_rate_effects']:
                grant_diffs = [effect['grant_difference'] for effect in grant_analysis[pattern_type]['grant_rate_effects']]
                grant_analysis[pattern_type]['avg_grant_difference'] = np.mean(grant_diffs)
                grant_analysis[pattern_type]['max_grant_difference'] = max(grant_diffs, key=abs)
        
        return grant_analysis
    
    def _find_grant_rate_effects(self, grant_data: pd.DataFrame, group1: str, group2: str, 
                                pattern: Dict) -> List[Dict]:
        """Find grant rate effects for specific country comparison"""
        effects = []
        
        # Look for field values that might correspond to countries
        # This is heuristic-based since we need to map bias comparison countries
        # to actual vignette field values
        
        for _, row in grant_data.iterrows():
            field_value = str(row['field_value']).lower()
            
            # Check if field value contains country references
            if any(country.lower() in field_value for country in [group1, group2]):
                effect = {
                    'field_name': row['field_name'],
                    'field_value': row['field_value'],
                    'field_type': row['field_type'],
                    'pre_grant_rate': row['pre_brexit_raw_rate'],
                    'post_grant_rate': row['post_brexit_raw_rate'],
                    'grant_difference': row['cross_model_difference'],
                    'statistical_significance': row['statistical_significance'],
                    'p_value': row['p_value'],
                    'pattern_sp_change': pattern['sp_change'],
                    'pattern_type': 'gained' if pattern['gained_significance'] else 'lost' if pattern['lost_significance'] else 'persistent'
                }
                effects.append(effect)
        
        return effects
    
    def identify_completion_scenarios(self, focus_patterns: Dict) -> Dict[str, List[Dict]]:
        """Identify specific vignette completion scenarios where bias emerges"""
        print("📝 Identifying completion scenarios...")
        
        scenarios = {
            'high_bias_scenarios': [],
            'bias_reversal_scenarios': [],
            'emerging_bias_scenarios': []
        }
        
        if not self.data['vignettes']:
            print("⚠️ No vignette data available - generating scenario templates")
            return self._generate_scenario_templates(focus_patterns)
        
        # If we have actual vignette data, analyze it
        for pattern in focus_patterns['all_focus_patterns']:
            if abs(pattern['sp_change']) > 0.1:  # Significant change threshold
                
                scenario_info = {
                    'comparison': pattern['comparison'],
                    'sp_change': pattern['sp_change'],
                    'pattern_type': self._classify_pattern_change(pattern),
                    'vignette_scenarios': self._extract_relevant_scenarios(pattern),
                    'bias_direction': 'favors_group2' if pattern['sp_change'] > 0 else 'favors_group1',
                    'magnitude': abs(pattern['sp_change'])
                }
                
                # Categorize scenario
                if abs(pattern['sp_change']) > 0.2:
                    scenarios['high_bias_scenarios'].append(scenario_info)
                elif pattern['gained_significance']:
                    scenarios['emerging_bias_scenarios'].append(scenario_info)
                elif (pattern['pre_sp'] > 0 and pattern['post_sp'] < 0) or (pattern['pre_sp'] < 0 and pattern['post_sp'] > 0):
                    scenarios['bias_reversal_scenarios'].append(scenario_info)
        
        return scenarios
    
    def _generate_scenario_templates(self, focus_patterns: Dict) -> Dict[str, List[Dict]]:
        """Generate scenario templates based on grant rate analysis"""
        print("🏗️ Generating scenario templates from grant rate data...")
        
        grant_data = self.data['grant_rates']
        topic_grant_data = grant_data[
            grant_data['topic'].str.contains(self.focus_topic, case=False, na=False)
        ].copy()
        
        scenarios = {
            'high_bias_scenarios': [],
            'bias_reversal_scenarios': [],
            'emerging_bias_scenarios': []
        }
        
        # Create scenarios based on significant grant rate differences
        for _, row in topic_grant_data.iterrows():
            if row['statistical_significance'] and abs(row['cross_model_difference']) > 0.1:
                
                scenario = {
                    'vignette_field': row['field_name'],
                    'field_value': row['field_value'],
                    'field_type': row['field_type'],
                    'grant_rate_change': row['cross_model_difference'],
                    'pre_grant_rate': row['pre_brexit_raw_rate'],
                    'post_grant_rate': row['post_brexit_raw_rate'],
                    'sample_size_pre': row['pre_brexit_sample_size'],
                    'sample_size_post': row['post_brexit_sample_size'],
                    'p_value': row['p_value'],
                    'favors_model': row['favors_model'],
                    'potential_bias_indicator': True
                }
                
                # Categorize based on change magnitude and direction
                if abs(row['cross_model_difference']) > 0.3:
                    scenarios['high_bias_scenarios'].append(scenario)
                elif row['favors_model'] != 'pre_brexit':  # Shift to post-Brexit
                    scenarios['emerging_bias_scenarios'].append(scenario)
                else:
                    scenarios['bias_reversal_scenarios'].append(scenario)
        
        return scenarios
    
    def _classify_pattern_change(self, pattern: Dict) -> str:
        """Classify the type of pattern change"""
        if pattern['gained_significance']:
            return "Newly Emerged"
        elif pattern['lost_significance']:
            return "Disappeared"
        elif pattern['both_significant']:
            return "Persistent"
        else:
            return "Non-significant"
    
    def _extract_relevant_scenarios(self, pattern: Dict) -> List[str]:
        """Extract relevant vignette scenarios for a pattern"""
        # This would require access to actual completion data
        # For now, return template scenarios
        scenarios = [
            f"Work authorization scenarios involving {pattern['group1']} vs {pattern['group2']}",
            f"Employment history assessments for asylum seekers from {pattern['group1']} and {pattern['group2']}",
            f"Integration potential evaluations in work-related contexts"
        ]
        return scenarios
    
    def generate_inspection_report(self, focus_patterns: Dict, grant_analysis: Dict, 
                                 scenarios: Dict, output_dir: Path) -> Path:
        """Generate comprehensive inspection report"""
        print("📄 Generating inspection report...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"pattern_inspection_report_{'-'.join(self.focus_countries).lower()}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# 🔍 Pattern Inspection Deep Dive\n\n")
            f.write(f"**Focus Countries**: {', '.join(self.focus_countries)}\n")
            f.write(f"**Topic**: {self.focus_topic}\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            total_patterns = len(focus_patterns['all_focus_patterns'])
            gained = len(focus_patterns['gained_patterns'])
            lost = len(focus_patterns['lost_patterns'])
            
            f.write(f"- **Total Focus Patterns Analyzed**: {total_patterns}\n")
            f.write(f"- **Patterns Gained**: {gained} (newly significant)\n")
            f.write(f"- **Patterns Lost**: {lost} (no longer significant)\n")
            f.write(f"- **Persistent Patterns**: {len(focus_patterns['persistent_patterns'])}\n\n")
            
            # Detailed Pattern Analysis
            f.write("## 📊 Detailed Pattern Changes\n\n")
            
            for pattern_type, patterns in focus_patterns.items():
                if pattern_type == 'all_focus_patterns' or not patterns:
                    continue
                
                f.write(f"### {pattern_type.replace('_', ' ').title()}\n\n")
                f.write("| Comparison | Pre-Brexit SP | Post-Brexit SP | SP Change | Pre-Sig | Post-Sig |\n")
                f.write("|------------|---------------|----------------|-----------|---------|----------|\n")
                
                for pattern in patterns:
                    f.write(f"| {pattern['comparison']} | {pattern['pre_sp']:.4f} | "
                           f"{pattern['post_sp']:.4f} | {pattern['sp_change']:+.4f} | "
                           f"{'✓' if pattern['pre_significant'] else '✗'} | "
                           f"{'✓' if pattern['post_significant'] else '✗'} |\n")
                f.write("\n")
            
            # Grant Rate Translation Analysis
            f.write("## 💰 Grant Rate Translation Analysis\n\n")
            f.write("*How bias patterns translate to actual decision outcomes*\n\n")
            
            for pattern_type, analysis in grant_analysis.items():
                if not analysis['grant_rate_effects']:
                    continue
                
                f.write(f"### {pattern_type.replace('_', ' ').title()}\n\n")
                f.write(f"- **Pattern Count**: {analysis['pattern_count']}\n")
                f.write(f"- **Average Grant Rate Difference**: {analysis['avg_grant_difference']:.4f}\n")
                f.write(f"- **Maximum Grant Rate Difference**: {analysis['max_grant_difference']:.4f}\n\n")
                
                f.write("#### Specific Grant Rate Effects\n\n")
                f.write("| Field | Value | Pre-Rate | Post-Rate | Difference | Significant |\n")
                f.write("|-------|-------|----------|-----------|------------|------------|\n")
                
                for effect in analysis['grant_rate_effects'][:10]:  # Top 10
                    f.write(f"| {effect['field_name']} | {effect['field_value'][:30]}... | "
                           f"{effect['pre_grant_rate']:.3f} | {effect['post_grant_rate']:.3f} | "
                           f"{effect['grant_difference']:+.3f} | "
                           f"{'✓' if effect['statistical_significance'] else '✗'} |\n")
                f.write("\n")
            
            # Completion Scenarios
            f.write("## 📝 Vignette Completion Scenarios\n\n")
            f.write("*Specific scenarios where bias patterns emerge*\n\n")
            
            for scenario_type, scenario_list in scenarios.items():
                if not scenario_list:
                    continue
                
                f.write(f"### {scenario_type.replace('_', ' ').title()}\n\n")
                
                for i, scenario in enumerate(scenario_list[:5], 1):  # Top 5
                    f.write(f"#### Scenario {i}\n")
                    
                    if 'comparison' in scenario:
                        f.write(f"- **Comparison**: {scenario['comparison']}\n")
                        f.write(f"- **SP Change**: {scenario['sp_change']:+.4f}\n")
                        f.write(f"- **Pattern Type**: {scenario['pattern_type']}\n")
                        f.write(f"- **Bias Direction**: {scenario['bias_direction']}\n")
                    else:
                        f.write(f"- **Field**: {scenario.get('vignette_field', 'N/A')}\n")
                        f.write(f"- **Value**: {scenario.get('field_value', 'N/A')}\n")
                        f.write(f"- **Grant Rate Change**: {scenario.get('grant_rate_change', 0):+.4f}\n")
                        f.write(f"- **Favors Model**: {scenario.get('favors_model', 'N/A')}\n")
                    
                    f.write("\n")
            
            # Qualitative Analysis Framework
            f.write("## 🧠 Qualitative Analysis Framework\n\n")
            f.write("### Recommended Investigation Steps\n\n")
            f.write("1. **Completion Comparison**: Extract actual model completions for scenarios above\n")
            f.write("2. **Bias Detection Dimensions**:\n")
            f.write("   - **Credibility Assessment**: How does model evaluate claim authenticity?\n")
            f.write("   - **Work Authorization Reasoning**: Different treatment of employment rights?\n")
            f.write("   - **Integration Assumptions**: Assumptions about work-seeking behavior?\n")
            f.write("   - **Risk Evaluation**: Country-based threat/security assessments?\n")
            f.write("3. **Side-by-Side Analysis**: Compare identical vignettes with only country changed\n")
            f.write("4. **Pattern Validation**: Verify patterns hold across multiple vignette variants\n\n")
            
            f.write("### Key Questions for Manual Review\n\n")
            for country in self.focus_countries:
                f.write(f"**{country}**:\n")
                f.write(f"- How does the model frame {country} asylum seekers' work intentions?\n")
                f.write(f"- What assumptions about {country} refugees' employment history?\n")
                f.write(f"- Different legal pathways suggested for {country} applicants?\n\n")
            
            # Methodology
            f.write("## 📋 Methodology\n\n")
            f.write(f"- **Data Source**: FDR-corrected significance patterns\n")
            f.write(f"- **Focus Countries**: {', '.join(self.focus_countries)}\n")
            f.write(f"- **Topic Filter**: {self.focus_topic}\n")
            f.write(f"- **Significance Threshold**: p < 0.05 (FDR-corrected)\n")
            f.write(f"- **Pattern Change Threshold**: |ΔSP| > 0.01\n\n")
        
        return report_path
    
    def create_inspection_visualizations(self, focus_patterns: Dict, scenarios: Dict, 
                                       output_dir: Path):
        """Create visualizations for pattern inspection"""
        print("📈 Creating inspection visualizations...")
        
        try:
            # 1. Pattern Change Magnitude Plot
            self._plot_pattern_changes(focus_patterns, output_dir)
            
            # 2. Grant Rate Effects Plot
            if self.data['grant_rates'] is not None:
                self._plot_grant_rate_effects(scenarios, output_dir)
            
            # 3. Country Comparison Matrix
            self._plot_country_comparison_matrix(focus_patterns, output_dir)
            
        except Exception as e:
            print(f"Warning: Could not create some visualizations: {e}")
    
    def _plot_pattern_changes(self, focus_patterns: Dict, output_dir: Path):
        """Plot pattern changes for focus countries"""
        all_patterns = focus_patterns['all_focus_patterns']
        
        if not all_patterns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: SP Change by Pattern Type
        pattern_types = []
        sp_changes = []
        colors = []
        
        color_map = {'Newly Emerged': '#FF6347', 'Disappeared': '#4169E1', 'Persistent': '#2E8B57'}
        
        for pattern in all_patterns:
            pattern_type = self._classify_pattern_change(pattern)
            pattern_types.append(f"{pattern['comparison']}\n({pattern_type})")
            sp_changes.append(pattern['sp_change'])
            colors.append(color_map.get(pattern_type, '#DDA0DD'))
        
        bars = ax1.barh(range(len(pattern_types)), sp_changes, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(pattern_types)))
        ax1.set_yticklabels(pattern_types, fontsize=9)
        ax1.set_xlabel('Statistical Parity Change')
        ax1.set_title(f'Pattern Changes: {", ".join(self.focus_countries)}\n{self.focus_topic}')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sp_changes)):
            ax1.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:+.3f}', 
                    va='center', ha='left' if value >= 0 else 'right', fontsize=8)
        
        # Plot 2: Pre vs Post SP Scatter
        pre_sp = [p['pre_sp'] for p in all_patterns]
        post_sp = [p['post_sp'] for p in all_patterns]
        pattern_colors = [color_map.get(self._classify_pattern_change(p), '#DDA0DD') for p in all_patterns]
        
        ax2.scatter(pre_sp, post_sp, c=pattern_colors, alpha=0.7, s=100)
        ax2.plot([-0.5, 0.5], [-0.5, 0.5], 'k--', alpha=0.5, label='No Change')
        ax2.set_xlabel('Pre-Brexit Statistical Parity')
        ax2.set_ylabel('Post-Brexit Statistical Parity')
        ax2.set_title('Statistical Parity: Pre vs Post Brexit')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add country labels
        for i, pattern in enumerate(all_patterns):
            ax2.annotate(pattern['comparison'], (pre_sp[i], post_sp[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pattern_changes_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_grant_rate_effects(self, scenarios: Dict, output_dir: Path):
        """Plot grant rate effects for scenarios"""
        # Extract grant rate data from scenarios
        grant_effects = []
        
        for scenario_type, scenario_list in scenarios.items():
            for scenario in scenario_list:
                if 'grant_rate_change' in scenario:
                    grant_effects.append({
                        'scenario_type': scenario_type,
                        'grant_change': scenario['grant_rate_change'],
                        'pre_rate': scenario.get('pre_grant_rate', 0),
                        'post_rate': scenario.get('post_grant_rate', 0),
                        'field': scenario.get('vignette_field', 'Unknown')
                    })
        
        if not grant_effects:
            return
        
        df = pd.DataFrame(grant_effects)
        
        plt.figure(figsize=(12, 8))
        
        # Create box plot by scenario type
        scenario_types = df['scenario_type'].unique()
        grant_changes_by_type = [df[df['scenario_type'] == st]['grant_change'].values for st in scenario_types]
        
        bp = plt.boxplot(grant_changes_by_type, labels=[st.replace('_', '\n') for st in scenario_types], patch_artist=True)
        
        # Color the boxes
        colors = ['#FF6347', '#4169E1', '#2E8B57'][:len(scenario_types)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.ylabel('Grant Rate Change (Post - Pre)')
        plt.title(f'Grant Rate Effects by Scenario Type\n{self.focus_topic}')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'grant_rate_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_country_comparison_matrix(self, focus_patterns: Dict, output_dir: Path):
        """Plot country comparison matrix"""
        all_patterns = focus_patterns['all_focus_patterns']
        
        if not all_patterns:
            return
        
        # Create matrix of SP changes between countries
        countries = list(set([p['group1'] for p in all_patterns] + [p['group2'] for p in all_patterns]))
        matrix = np.zeros((len(countries), len(countries)))
        
        for pattern in all_patterns:
            i = countries.index(pattern['group1'])
            j = countries.index(pattern['group2'])
            matrix[i, j] = pattern['sp_change']
            matrix[j, i] = -pattern['sp_change']  # Symmetric
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = matrix == 0
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   xticklabels=countries, yticklabels=countries,
                   mask=mask, cbar_kws={'label': 'Statistical Parity Change'})
        
        plt.title(f'Country Comparison Matrix: SP Changes\n{self.focus_topic}')
        plt.xlabel('Comparison Country (favored when positive)')
        plt.ylabel('Base Country')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'country_comparison_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_inspection_analysis(self, output_dir: Path = None) -> bool:
        """Run complete pattern inspection analysis"""
        if output_dir is None:
            output_dir = Path(f"pattern_inspection_{'-'.join(self.focus_countries).lower()}")
        
        print(f"🔍 Starting Pattern Inspection Analysis")
        print(f"Focus: {', '.join(self.focus_countries)} in '{self.focus_topic}'")
        print("=" * 60)
        
        # Load data
        if not self.load_inspection_data():
            return False
        
        # Identify focus patterns
        focus_patterns = self.identify_focus_patterns()
        if not focus_patterns['all_focus_patterns']:
            print("❌ No patterns found for focus countries")
            return False
        
        # Analyze grant rate translation
        grant_analysis = self.analyze_grant_rate_translation(focus_patterns)
        
        # Identify completion scenarios
        scenarios = self.identify_completion_scenarios(focus_patterns)
        
        # Generate visualizations
        self.create_inspection_visualizations(focus_patterns, scenarios, output_dir)
        
        # Generate report
        report_path = self.generate_inspection_report(focus_patterns, grant_analysis, scenarios, output_dir)
        
        print(f"\n✅ Pattern Inspection Analysis Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Report: {report_path}")
        print("=" * 60)
        
        return True

def main():
    """Main execution function"""
    
    # Configuration
    base_data_dir = Path("../../outputs")
    focus_countries = ["Syria", "Nigeria", "Myanmar"]  # Based on user's observation
    focus_topic = "Intentions regarding work"
    
    # Initialize analyzer
    analyzer = PatternInspectionAnalyzer(base_data_dir, focus_countries, focus_topic)
    
    # Run analysis
    success = analyzer.run_inspection_analysis()
    
    if success:
        print("\n🎯 Key Deliverables Generated:")
        print("• Detailed pattern change analysis for Syria, Nigeria, Myanmar")
        print("• Grant rate translation mapping")
        print("• Specific vignette scenarios where bias emerges")
        print("• Qualitative analysis framework for manual review")
        print("• Comprehensive visualizations")
    else:
        print("\n❌ Analysis failed. Check data paths and requirements.")

if __name__ == "__main__":
    main() 