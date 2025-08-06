#!/usr/bin/env python3
"""
🎯 TOPIC DEEP DIVE ANALYZER

A robust, reconfigurable tool for analyzing bias pattern changes across topics.
Examines:
- Significant bias pattern transitions (newly emerged, persistent, disappeared)
- Granular analysis across sub-topics/completions  
- Analysis across all protected attributes (country, age, religion, gender)
- FDR-corrected significance patterns

Usage:
    python topic_analyzer.py --topic "Intentions regarding work"
    python topic_analyzer.py --topic "Intentions regarding education" --output-dir custom_output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class BiasPattern:
    """Represents a bias pattern with metadata"""
    comparison: str
    topic: str
    attribute: str
    pre_magnitude: float
    post_magnitude: float
    magnitude_change: float
    pre_significant: bool
    post_significant: bool
    pattern_type: str  # NEWLY_EMERGED, PERSISTENT, DISAPPEARED
    favored_group: str
    disfavored_group: str

@dataclass
class TopicAnalysisConfig:
    """Configuration for topic analysis"""
    topic_filter: str
    base_data_dir: Path
    output_dir: Path
    include_subtopics: bool = True
    fdr_correction: bool = True
    min_magnitude_threshold: float = 0.0
    
class TopicDeepDiveAnalyzer:
    """Comprehensive analyzer for topic-specific bias patterns"""
    
    def __init__(self, config: TopicAnalysisConfig):
        self.config = config
        self.data = {}
        self.patterns = []
        self.analysis_results = {}
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load all relevant datasets for analysis"""
        base_dir = self.config.base_data_dir
        
        try:
            # Core fairness comparisons data
            self.data['fairness_comparisons'] = pd.read_csv(
                base_dir / "vector_drift" / "deduplicated_fairness_comparisons.csv"
            )
            
            # Country-specific analysis data  
            self.data['detailed_transitions'] = pd.read_csv(
                base_dir / "country_analysis" / "detailed_pattern_transitions.csv"
            )
            
            self.data['country_significant'] = pd.read_csv(
                base_dir / "country_analysis" / "significant_country_comparisons_fdr.csv"
            )
            
            # Cross-attribute analysis
            self.data['topic_attribute_data'] = pd.read_csv(
                base_dir / "visualizations" / "topic_attribute_analysis_data.csv"
            )
            
            # Gender analysis if available
            try:
                self.data['gender_bias'] = pd.read_csv(
                    base_dir / "gender_analysis" / "persistent_gender_bias_detailed.csv"
                )
            except FileNotFoundError:
                self.data['gender_bias'] = pd.DataFrame()
                
            # Normative analysis for sub-topic granularity
            try:
                self.data['grant_rate_analysis'] = pd.read_csv(
                    base_dir.parent.parent / "normative" / "outputs" / "grant_rate_analysis" / 
                    "grant_rate_analysis_by_vignette_fields_enhanced_FIXED.csv"
                )
            except FileNotFoundError:
                self.data['grant_rate_analysis'] = pd.DataFrame()
                
            print(f"✅ Loaded {len(self.data)} datasets successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def filter_topic_data(self) -> Dict[str, pd.DataFrame]:
        """Filter all datasets for the specified topic"""
        topic_filter = self.config.topic_filter
        filtered_data = {}
        
        # Filter fairness comparisons
        mask = self.data['fairness_comparisons']['comparison_label'].str.contains(
            topic_filter, case=False, na=False
        )
        filtered_data['fairness_comparisons'] = self.data['fairness_comparisons'][mask].copy()
        
        # Filter detailed transitions
        mask = self.data['detailed_transitions']['topic'].str.contains(
            topic_filter, case=False, na=False
        )
        filtered_data['detailed_transitions'] = self.data['detailed_transitions'][mask].copy()
        
        # Filter significant country comparisons
        mask = self.data['country_significant']['topic'].str.contains(
            topic_filter, case=False, na=False
        )
        filtered_data['country_significant'] = self.data['country_significant'][mask].copy()
        
        # Filter topic-attribute data
        mask = self.data['topic_attribute_data']['topic'].str.contains(
            topic_filter, case=False, na=False
        )
        filtered_data['topic_attribute_data'] = self.data['topic_attribute_data'][mask].copy()
        
        # Filter gender bias data
        if not self.data['gender_bias'].empty:
            mask = self.data['gender_bias']['topic'].str.contains(
                topic_filter, case=False, na=False
            )
            filtered_data['gender_bias'] = self.data['gender_bias'][mask].copy()
        else:
            filtered_data['gender_bias'] = pd.DataFrame()
            
        # Filter grant rate analysis for sub-topics
        if not self.data['grant_rate_analysis'].empty:
            mask = self.data['grant_rate_analysis']['topic'].str.contains(
                topic_filter, case=False, na=False
            )
            filtered_data['grant_rate_analysis'] = self.data['grant_rate_analysis'][mask].copy()
        else:
            filtered_data['grant_rate_analysis'] = pd.DataFrame()
            
        return filtered_data
    
    def analyze_pattern_transitions(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze bias pattern transitions for the topic"""
        transitions_df = filtered_data['detailed_transitions']
        
        if transitions_df.empty:
            return {"error": "No transition data found for topic"}
        
        # Pattern type distribution
        pattern_counts = transitions_df['pattern_type'].value_counts()
        
        # Calculate net change
        emerged = pattern_counts.get('NEWLY_EMERGED', 0)
        disappeared = pattern_counts.get('DISAPPEARED', 0) 
        persistent = pattern_counts.get('PERSISTENT', 0)
        net_change = emerged - disappeared
        
        # Analyze magnitude changes
        magnitude_stats = {
            'mean_magnitude_change': transitions_df['abs_sp_change'].mean(),
            'max_magnitude_change': transitions_df['abs_sp_change'].max(),
            'min_magnitude_change': transitions_df['abs_sp_change'].min()
        }
        
        # Country impact analysis
        country_impact = {}
        for country in pd.concat([transitions_df['country1'], transitions_df['country2']]).unique():
            country_patterns = transitions_df[
                (transitions_df['country1'] == country) | (transitions_df['country2'] == country)
            ]
            
            lost_patterns = len(country_patterns[country_patterns['pattern_type'] == 'DISAPPEARED'])
            gained_patterns = len(country_patterns[country_patterns['pattern_type'] == 'NEWLY_EMERGED'])
            
            country_impact[country] = {
                'lost_patterns': lost_patterns,
                'gained_patterns': gained_patterns,
                'net_change': gained_patterns - lost_patterns
            }
        
        return {
            'pattern_distribution': pattern_counts.to_dict(),
            'net_change': net_change,
            'magnitude_stats': magnitude_stats,
            'country_impact': country_impact,
            'total_patterns': len(transitions_df)
        }
    
    def analyze_cross_attribute_patterns(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze patterns across different protected attributes"""
        attr_data = filtered_data['topic_attribute_data']
        
        if attr_data.empty:
            return {"error": "No cross-attribute data found"}
        
        # Analyze by attribute type
        attribute_analysis = {}
        
        for attribute in attr_data['attribute'].unique():
            attr_subset = attr_data[attr_data['attribute'] == attribute].iloc[0]
            
            attribute_analysis[attribute] = {
                'pre_patterns': int(attr_subset['pre_significant']),
                'post_patterns': int(attr_subset['post_significant']),
                'persistent_patterns': 0,  # Will calculate from other columns
                'net_change': int(attr_subset['significance_change']),
                'pre_magnitude_mean': float(attr_subset['pre_brexit_bias']),
                'post_magnitude_mean': float(attr_subset['post_brexit_bias'])
            }
        
        return attribute_analysis
    
    def analyze_subtopic_granularity(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze granular patterns within topic sub-categories"""
        grant_data = filtered_data['grant_rate_analysis']
        
        if grant_data.empty:
            return {"message": "No granular sub-topic data available"}
        
        # Group by field type to see different aspects
        subtopic_analysis = {}
        
        for field_type in grant_data['field_type'].unique():
            field_subset = grant_data[grant_data['field_type'] == field_type]
            
            subtopic_analysis[field_type] = {
                'total_variations': len(field_subset),
                'variations': []
            }
            
            for _, row in field_subset.iterrows():
                variation_data = {
                    'value': row['field_value'],
                    'pre_magnitude': float(row['pre_brexit_normalized']),
                    'post_magnitude': float(row['post_brexit_normalized']),
                    'magnitude_change': float(row['cross_model_difference']),
                    'pre_grant_rate': float(row.get('pre_brexit_raw_rate', 0)),
                    'post_grant_rate': float(row.get('post_brexit_raw_rate', 0))
                }
                subtopic_analysis[field_type]['variations'].append(variation_data)
        
        return subtopic_analysis

    def analyze_intersectional_group_disadvantage(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        A. INTERSECTIONAL GROUP DISADVANTAGE ANALYSIS
        
        Identify which groups face the strongest negative disparities in this topic
        and whether those disparities are statistically significant.
        """
        print("🧠 Analyzing intersectional group disadvantage...")
        
        fairness_data = filtered_data['fairness_comparisons']
        
        # Parse group comparisons to extract individual groups
        group_disadvantage = []
        
        for _, row in fairness_data.iterrows():
            label = row['comparison_label']
            
            # Parse: "Group1_vs_Group2 (attribute) [topic]"
            if '(' in label and ')' in label and '[' in label:
                main_part = label.split(' (')[0]
                remainder = label.split(' (')[1]
                attribute = remainder.split(')')[0]
                
                if '_vs_' in main_part:
                    group1, group2 = main_part.split('_vs_')
                    
                    # Pre-Brexit disadvantage
                    pre_sp = row['pre_brexit_sp_magnitude']
                    post_sp = row['post_brexit_sp_magnitude']
                    pre_sig = row['pre_brexit_sp_significance']
                    post_sig = row['post_brexit_sp_significance']
                    
                    # Group1 perspective (negative SP means Group1 disadvantaged)
                    if pre_sig or post_sig:  # Only include FDR-significant patterns
                        group_disadvantage.append({
                            'group': group1,
                            'comparison_group': group2,
                            'attribute': attribute,
                            'pre_disadvantage': -pre_sp if pre_sp < 0 else 0,
                            'post_disadvantage': -post_sp if post_sp < 0 else 0,
                            'pre_significant': pre_sig,
                            'post_significant': post_sig,
                            'pattern_type': self._classify_pattern_transition(pre_sig, post_sig),
                            'comparison_label': label
                        })
                        
                        # Group2 perspective (positive SP means Group2 disadvantaged relative to Group1)
                        group_disadvantage.append({
                            'group': group2,
                            'comparison_group': group1,
                            'attribute': attribute,
                            'pre_disadvantage': pre_sp if pre_sp > 0 else 0,
                            'post_disadvantage': post_sp if post_sp > 0 else 0,
                            'pre_significant': pre_sig,
                            'post_significant': post_sig,
                            'pattern_type': self._classify_pattern_transition(pre_sig, post_sig),
                            'comparison_label': label
                        })
        
        # Convert to DataFrame for analysis
        disadvantage_df = pd.DataFrame(group_disadvantage)
        
        if disadvantage_df.empty:
            return {"message": "No significant group disadvantages found"}
        
        # Filter for meaningful disadvantage (> 0)
        disadvantaged = disadvantage_df[
            (disadvantage_df['pre_disadvantage'] > 0) | 
            (disadvantage_df['post_disadvantage'] > 0)
        ].copy()
        
        # Rank by maximum disadvantage across both eras
        disadvantaged['max_disadvantage'] = disadvantaged[['pre_disadvantage', 'post_disadvantage']].max(axis=1)
        disadvantaged_ranked = disadvantaged.sort_values('max_disadvantage', ascending=False)
        
        # Group-level analysis
        group_summary = disadvantaged_ranked.groupby(['group', 'attribute']).agg({
            'max_disadvantage': 'max',
            'pre_disadvantage': 'mean',
            'post_disadvantage': 'mean',
            'comparison_label': 'count'
        }).rename(columns={'comparison_label': 'comparison_count'}).reset_index()
        
        # Intersectional analysis - identify groups appearing across multiple attributes
        intersectional_groups = {}
        for group in group_summary['group'].unique():
            group_attrs = group_summary[group_summary['group'] == group]
            if len(group_attrs) > 1:  # Group appears in multiple attribute comparisons
                intersectional_groups[group] = {
                    'attributes': group_attrs['attribute'].tolist(),
                    'total_disadvantage': group_attrs['max_disadvantage'].sum(),
                    'avg_disadvantage': group_attrs['max_disadvantage'].mean(),
                    'comparison_count': group_attrs['comparison_count'].sum()
                }
        
        return {
            'top_disadvantaged_groups': disadvantaged_ranked.head(10).to_dict('records'),
            'group_summary': group_summary.to_dict('records'),
            'intersectional_analysis': intersectional_groups,
            'total_significant_disadvantages': len(disadvantaged),
            'attributes_analyzed': disadvantaged['attribute'].unique().tolist()
        }
    
    def analyze_pattern_persistence_shifts(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        B. PATTERN PERSISTENCE/SHIFT ANALYSIS
        
        Understand which group-level disparities are stable, emerged, or disappeared
        between pre- and post-Brexit models.
        """
        print("🔄 Analyzing pattern persistence and shifts...")
        
        fairness_data = filtered_data['fairness_comparisons']
        
        # Build comparison matrix
        pattern_transitions = []
        
        for _, row in fairness_data.iterrows():
            label = row['comparison_label']
            
            if '(' in label and ')' in label and '[' in label:
                main_part = label.split(' (')[0]
                remainder = label.split(' (')[1]
                attribute = remainder.split(')')[0]
                
                if '_vs_' in main_part:
                    group1, group2 = main_part.split('_vs_')
                    
                    pre_sp = row['pre_brexit_sp_magnitude']
                    post_sp = row['post_brexit_sp_magnitude']
                    sp_change = row['sp_magnitude_difference']
                    pre_sig = row['pre_brexit_sp_significance']
                    post_sig = row['post_brexit_sp_significance']
                    
                    # Classify pattern transition
                    if pre_sig and post_sig:
                        transition = "Persistent_Bias"
                    elif not pre_sig and post_sig:
                        transition = "Emergent_Bias"
                    elif pre_sig and not post_sig:
                        transition = "Disappeared_Bias"
                    else:
                        transition = "Non_Significant"
                    
                    # Classify bias direction change
                    direction_change = "Stable"
                    if (pre_sp > 0 and post_sp < 0) or (pre_sp < 0 and post_sp > 0):
                        direction_change = "Bias_Reversal"
                    elif abs(sp_change) > 0.1:  # Significant magnitude change
                        if abs(post_sp) > abs(pre_sp):
                            direction_change = "Bias_Amplification"
                        else:
                            direction_change = "Bias_Reduction"
                    
                    pattern_transitions.append({
                        'comparison': f"{group1} vs {group2}",
                        'attribute': attribute,
                        'pre_brexit_sp': pre_sp,
                        'post_brexit_sp': post_sp,
                        'sp_change': sp_change,
                        'transition_type': transition,
                        'direction_change': direction_change,
                        'pre_significant': pre_sig,
                        'post_significant': post_sig,
                        'magnitude_change': abs(sp_change)
                    })
        
        transitions_df = pd.DataFrame(pattern_transitions)
        
        if transitions_df.empty:
            return {"message": "No pattern transitions found"}
        
        # Analysis by transition type
        transition_summary = transitions_df['transition_type'].value_counts().to_dict()
        direction_summary = transitions_df['direction_change'].value_counts().to_dict()
        
        # Most dramatic changes
        dramatic_changes = transitions_df.nlargest(10, 'magnitude_change')[
            ['comparison', 'attribute', 'pre_brexit_sp', 'post_brexit_sp', 
             'sp_change', 'transition_type', 'direction_change']
        ].to_dict('records')
        
        # Attribute-wise analysis
        attr_analysis = {}
        for attr in transitions_df['attribute'].unique():
            attr_data = transitions_df[transitions_df['attribute'] == attr]
            attr_analysis[attr] = {
                'total_comparisons': len(attr_data),
                'transition_types': attr_data['transition_type'].value_counts().to_dict(),
                'avg_magnitude_change': attr_data['magnitude_change'].mean(),
                'max_change': attr_data['magnitude_change'].max()
            }
        
        return {
            'transition_summary': transition_summary,
            'direction_summary': direction_summary,
            'dramatic_changes': dramatic_changes,
            'attribute_analysis': attr_analysis,
            'pattern_matrix': transitions_df.to_dict('records')
        }
    
    def analyze_qualitative_vignette_completions(self, filtered_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        C. QUALITATIVE VIGNETTE COMPLETION ANALYSIS
        
        Move from metrics to meaning - show how bias manifests in text outputs.
        Note: This requires access to actual vignette completions/text outputs.
        """
        print("📝 Analyzing qualitative vignette completions...")
        
        # Get maximum divergence cases for detailed analysis
        fairness_data = filtered_data['fairness_comparisons']
        
        # Find cases with highest statistical parity changes
        max_divergence_cases = fairness_data.nlargest(5, 'sp_magnitude_difference')[
            ['comparison_label', 'pre_brexit_sp_magnitude', 'post_brexit_sp_magnitude', 
             'sp_magnitude_difference', 'pre_brexit_sp_significance', 'post_brexit_sp_significance']
        ].to_dict('records')
        
        # Parse for structured analysis
        analysis_targets = []
        for case in max_divergence_cases:
            label = case['comparison_label']
            if '(' in label and ')' in label and '[' in label:
                main_part = label.split(' (')[0]
                remainder = label.split(' (')[1]
                attribute = remainder.split(')')[0]
                topic = remainder.split('[')[1].split(']')[0]
                
                if '_vs_' in main_part:
                    group1, group2 = main_part.split('_vs_')
                    
                    analysis_targets.append({
                        'group_pair': f"{group1} vs {group2}",
                        'attribute': attribute,
                        'topic': topic,
                        'sp_change': case['sp_magnitude_difference'],
                        'pre_sp': case['pre_brexit_sp_magnitude'],
                        'post_sp': case['post_brexit_sp_magnitude'],
                        'vignette_template': topic,  # This would be expanded with actual vignette content
                        'recommended_analysis': self._generate_qualitative_analysis_framework(
                            group1, group2, attribute, case['sp_magnitude_difference']
                        )
                    })
        
        return {
            'max_divergence_cases': analysis_targets,
            'analysis_framework': {
                'tone_analysis': "Look for differences in emotional register and assumptions",
                'legal_reasoning': "Compare legal logic and admissibility reasoning",
                'moral_framing': "Analyze assumptions about motivation and authenticity",
                'integration_assumptions': "Check for different integration/threat assessments"
            },
            'suggested_method': "Side-by-side completion comparison with structured annotation",
            'note': "Requires access to actual model completion texts for full analysis"
        }
    
    def _classify_pattern_transition(self, pre_sig: bool, post_sig: bool) -> str:
        """Helper to classify pattern transitions"""
        if pre_sig and post_sig:
            return "Persistent"
        elif not pre_sig and post_sig:
            return "Newly_Emerged"
        elif pre_sig and not post_sig:
            return "Disappeared"
        else:
            return "Non_Significant"
    
    def _generate_qualitative_analysis_framework(self, group1: str, group2: str, 
                                               attribute: str, sp_change: float) -> Dict:
        """Generate structured framework for qualitative analysis"""
        direction = "favoring" if sp_change > 0 else "penalizing"
        
        return {
            'focus_question': f"How does the model treat {group1} vs {group2} differently in {attribute}-based reasoning?",
            'bias_direction': f"Model shifted toward {direction} {group2 if sp_change > 0 else group1}",
            'analysis_dimensions': [
                "Credibility assumptions",
                "Legal pathway reasoning", 
                "Integration potential assessment",
                "Risk/threat evaluation",
                "Emotional tone and empathy"
            ],
            'expected_differences': f"Look for {abs(sp_change):.3f} magnitude difference in treatment"
        }
    
    def generate_insights(self, analysis_results: Dict) -> List[str]:
        """Generate interpretative insights from analysis results"""
        insights = []
        
        # Pattern transition insights
        if 'pattern_transitions' in analysis_results:
            transitions = analysis_results['pattern_transitions']
            
            if 'net_change' in transitions:
                net_change = transitions['net_change']
                if net_change > 0:
                    insights.append(f"📈 **Net Increase**: {net_change} more significant bias patterns emerged than disappeared")
                elif net_change < 0:
                    insights.append(f"📉 **Net Decrease**: {abs(net_change)} more patterns disappeared than emerged")
                else:
                    insights.append("⚖️ **No Net Change**: Equal number of patterns emerged and disappeared")
            
            # Country impact insights
            if 'country_impact' in transitions:
                country_impacts = transitions['country_impact']
                
                # Find biggest losers and gainers
                biggest_loser = max(country_impacts.items(), key=lambda x: x[1]['lost_patterns'])
                biggest_gainer = max(country_impacts.items(), key=lambda x: x[1]['gained_patterns'])
                
                if biggest_loser[1]['lost_patterns'] > 0:
                    insights.append(f"🔻 **Major Pattern Loss**: {biggest_loser[0]} lost {biggest_loser[1]['lost_patterns']} significant bias patterns")
                
                if biggest_gainer[1]['gained_patterns'] > 0:
                    insights.append(f"🔺 **Major Pattern Gain**: {biggest_gainer[0]} gained {biggest_gainer[1]['gained_patterns']} new significant bias patterns")
        
        # Cross-attribute insights
        if 'cross_attribute_patterns' in analysis_results:
            attr_analysis = analysis_results['cross_attribute_patterns']
            
            # Find most impacted attribute
            attr_impacts = {attr: data['net_change'] for attr, data in attr_analysis.items() if isinstance(data, dict)}
            if attr_impacts:
                most_impacted = min(attr_impacts.items(), key=lambda x: x[1])
                insights.append(f"🎯 **Most Impacted Attribute**: {most_impacted[0]} (net change: {most_impacted[1]} patterns)")
        
        return insights
    
    def create_visualizations(self, filtered_data: Dict[str, pd.DataFrame], analysis_results: Dict):
        """Create comprehensive visualizations"""
        
        # 1. Pattern Transition Overview
        if 'pattern_transitions' in analysis_results:
            self._plot_pattern_transitions(analysis_results['pattern_transitions'])
        
        # 2. Cross-Attribute Comparison
        if 'cross_attribute_patterns' in analysis_results:
            self._plot_cross_attribute_patterns(analysis_results['cross_attribute_patterns'])
        
        # 3. Country Impact Heatmap
        if 'pattern_transitions' in analysis_results and 'country_impact' in analysis_results['pattern_transitions']:
            self._plot_country_impact(analysis_results['pattern_transitions']['country_impact'])
        
        # 4. Magnitude Change Distribution
        if 'fairness_comparisons' in filtered_data and not filtered_data['fairness_comparisons'].empty:
            self._plot_magnitude_distribution(filtered_data['fairness_comparisons'])
        
        # 5. NEW: Intersectional Group Disadvantage Chart
        if 'intersectional_disadvantage' in analysis_results:
            self._plot_intersectional_disadvantage(analysis_results['intersectional_disadvantage'])
        
        # 6. NEW: Pattern Persistence Matrix
        if 'pattern_persistence' in analysis_results:
            self._plot_pattern_persistence_matrix(analysis_results['pattern_persistence'])
    
    def _plot_pattern_transitions(self, transition_data: Dict):
        """Plot pattern transition overview"""
        if 'pattern_distribution' not in transition_data:
            return
            
        patterns = transition_data['pattern_distribution']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pattern type distribution
        pattern_types = list(patterns.keys())
        pattern_counts = list(patterns.values())
        colors = ['#e74c3c', '#f39c12', '#27ae60']  # Red, Orange, Green
        
        ax1.bar(pattern_types, pattern_counts, color=colors[:len(pattern_types)])
        ax1.set_title(f'Bias Pattern Transitions\n{self.config.topic_filter}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Patterns')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(pattern_counts):
            ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Net change visualization
        net_change = transition_data.get('net_change', 0)
        ax2.bar(['Net Change'], [net_change], 
                color='green' if net_change > 0 else 'red' if net_change < 0 else 'gray')
        ax2.set_title('Net Pattern Change', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Net Change in Patterns')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value label
        ax2.text(0, net_change + (0.1 if net_change >= 0 else -0.1), str(net_change), 
                ha='center', va='bottom' if net_change >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'pattern_transitions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_attribute_patterns(self, attr_data: Dict):
        """Plot cross-attribute pattern analysis"""
        if not attr_data or 'error' in attr_data:
            return
            
        attributes = list(attr_data.keys())
        pre_patterns = [attr_data[attr]['pre_patterns'] for attr in attributes]
        post_patterns = [attr_data[attr]['post_patterns'] for attr in attributes]
        net_changes = [attr_data[attr]['net_change'] for attr in attributes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pre vs Post patterns
        x = np.arange(len(attributes))
        width = 0.35
        
        ax1.bar(x - width/2, pre_patterns, width, label='Pre-Brexit', alpha=0.8, color='#3498db')
        ax1.bar(x + width/2, post_patterns, width, label='Post-Brexit', alpha=0.8, color='#e74c3c')
        
        ax1.set_title(f'Significant Patterns by Attribute\n{self.config.topic_filter}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Significant Patterns')
        ax1.set_xticks(x)
        ax1.set_xticklabels(attributes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Net change by attribute
        colors = ['green' if nc > 0 else 'red' if nc < 0 else 'gray' for nc in net_changes]
        bars = ax2.bar(attributes, net_changes, color=colors, alpha=0.7)
        ax2.set_title('Net Change by Attribute', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Net Change in Patterns')
        ax2.set_xticklabels(attributes, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, nc in zip(bars, net_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                    f'{nc}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'cross_attribute_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_country_impact(self, country_impact: Dict):
        """Plot country-level impact heatmap"""
        if not country_impact:
            return
            
        countries = list(country_impact.keys())
        lost_patterns = [country_impact[c]['lost_patterns'] for c in countries]
        gained_patterns = [country_impact[c]['gained_patterns'] for c in countries]
        
        # Create data matrix for heatmap
        data_matrix = np.array([lost_patterns, gained_patterns])
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(countries)))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(countries, rotation=45, ha='right')
        ax.set_yticklabels(['Patterns Lost', 'Patterns Gained'])
        
        # Add text annotations
        for i in range(2):
            for j in range(len(countries)):
                text = ax.text(j, i, data_matrix[i, j], ha="center", va="center", 
                             color="white" if data_matrix[i, j] > np.max(data_matrix)/2 else "black",
                             fontweight='bold')
        
        ax.set_title(f'Country-Level Pattern Impact\n{self.config.topic_filter}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of Patterns', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'country_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_magnitude_distribution(self, transitions_df: pd.DataFrame):
        """Plot magnitude change distribution"""
        try:
            # Filter for meaningful changes - detect correct column name
            if 'sp_magnitude_difference' in transitions_df.columns:
                magnitude_col = 'sp_magnitude_difference'
            elif 'sp_magnitude_change' in transitions_df.columns:
                magnitude_col = 'sp_magnitude_change'
            else:
                # Find any column with magnitude or change in the name
                magnitude_cols = transitions_df.columns[transitions_df.columns.str.contains('magnitude|change', case=False)]
                if len(magnitude_cols) > 0:
                    magnitude_col = magnitude_cols[0]
                else:
                    print("Warning: No magnitude change column found")
                    return
            
            meaningful_changes = transitions_df[transitions_df[magnitude_col].abs() > 0.01]
            
            if meaningful_changes.empty:
                return
            
            plt.figure(figsize=(10, 6))
            
            # Histogram of magnitude changes
            plt.hist(meaningful_changes[magnitude_col], bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Change')
            
            plt.title(f'Statistical Parity Change Distribution\n{self.config.topic_filter}')
            plt.xlabel('SP Magnitude Change')
            plt.ylabel('Number of Comparisons')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'magnitude_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create magnitude distribution plot: {e}")

    def _plot_intersectional_disadvantage(self, disadvantage_data: Dict):
        """Plot intersectional group disadvantage analysis"""
        try:
            if 'top_disadvantaged_groups' not in disadvantage_data:
                return
            
            groups_data = disadvantage_data['top_disadvantaged_groups'][:15]  # Top 15
            
            if not groups_data:
                return
            
            # Create DataFrame for plotting
            df = pd.DataFrame(groups_data)
            
            # Create horizontal bar chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
            
            # Plot 1: Top Disadvantaged Groups by Maximum Disadvantage
            df_sorted = df.sort_values('max_disadvantage', ascending=True)
            
            # Color by attribute
            colors = {'country': '#ff7f0e', 'religion': '#2ca02c', 'gender': '#d62728', 'age': '#9467bd'}
            bar_colors = [colors.get(attr, '#1f77b4') for attr in df_sorted['attribute']]
            
            bars = ax1.barh(range(len(df_sorted)), df_sorted['max_disadvantage'], color=bar_colors, alpha=0.8)
            ax1.set_yticks(range(len(df_sorted)))
            ax1.set_yticklabels([f"{row['group']} ({row['attribute']})" for _, row in df_sorted.iterrows()], fontsize=9)
            ax1.set_xlabel('Maximum Disadvantage (Statistical Parity)')
            ax1.set_title('Top 15 Most Disadvantaged Groups\nWork-Related Vignettes')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, df_sorted['max_disadvantage'])):
                ax1.text(value + 0.005, i, f'{value:.3f}', va='center', fontsize=8)
            
            # Create legend for attributes
            legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=attr.title()) 
                             for attr, color in colors.items()]
            ax1.legend(handles=legend_elements, loc='lower right')
            
            # Plot 2: Pattern Type Distribution
            if 'pattern_type' in df.columns:
                pattern_counts = df['pattern_type'].value_counts()
                
                wedges, texts, autotexts = ax2.pie(pattern_counts.values, labels=pattern_counts.index, 
                                                  autopct='%1.1f%%', startangle=90)
                ax2.set_title('Disadvantage Pattern Types\n(FDR-Significant Only)')
                
                # Make text more readable
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            
            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'intersectional_disadvantage.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create intersectional disadvantage plot: {e}")

    def _plot_pattern_persistence_matrix(self, persistence_data: Dict):
        """Plot pattern persistence and shift analysis"""
        try:
            if 'pattern_matrix' not in persistence_data:
                return
            
            df = pd.DataFrame(persistence_data['pattern_matrix'])
            
            if df.empty:
                return
            
            # Create complex analysis visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Transition Type Distribution
            if 'transition_summary' in persistence_data:
                transition_counts = persistence_data['transition_summary']
                
                bars = ax1.bar(range(len(transition_counts)), list(transition_counts.values()), 
                              color=['#2E8B57', '#FF6347', '#4169E1', '#DDA0DD'])
                ax1.set_xticks(range(len(transition_counts)))
                ax1.set_xticklabels([t.replace('_', '\n') for t in transition_counts.keys()], rotation=45, ha='right')
                ax1.set_ylabel('Number of Patterns')
                ax1.set_title('Bias Pattern Transitions\nPre-Brexit → Post-Brexit')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, transition_counts.values()):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, str(value), 
                            ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Direction Change Analysis
            if 'direction_summary' in persistence_data:
                direction_counts = persistence_data['direction_summary']
                
                colors_dir = ['#32CD32', '#FF4500', '#4682B4', '#DA70D6', '#20B2AA']
                ax2.pie(direction_counts.values(), labels=direction_counts.keys(), autopct='%1.1f%%', 
                       startangle=90, colors=colors_dir[:len(direction_counts)])
                ax2.set_title('Bias Direction Changes\n(Magnitude & Direction)')
            
            # Plot 3: Attribute-wise Transition Analysis
            if 'attribute_analysis' in persistence_data:
                attr_analysis = persistence_data['attribute_analysis']
                
                attributes = list(attr_analysis.keys())
                avg_changes = [data['avg_magnitude_change'] for data in attr_analysis.values()]
                max_changes = [data['max_change'] for data in attr_analysis.values()]
                
                x = np.arange(len(attributes))
                width = 0.35
                
                bars1 = ax3.bar(x - width/2, avg_changes, width, label='Average Change', alpha=0.8, color='skyblue')
                bars2 = ax3.bar(x + width/2, max_changes, width, label='Maximum Change', alpha=0.8, color='orange')
                
                ax3.set_xlabel('Protected Attribute')
                ax3.set_ylabel('Statistical Parity Change Magnitude')
                ax3.set_title('Change Magnitudes by Attribute')
                ax3.set_xticks(x)
                ax3.set_xticklabels(attributes)
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001, f'{height:.3f}', 
                                ha='center', va='bottom', fontsize=8)
            
            # Plot 4: Magnitude Change Scatter Plot
            if len(df) > 0:
                # Color by transition type
                transition_colors = {'Persistent_Bias': '#2E8B57', 'Emergent_Bias': '#FF6347', 
                                   'Disappeared_Bias': '#4169E1', 'Non_Significant': '#DDA0DD'}
                
                for transition_type in df['transition_type'].unique():
                    subset = df[df['transition_type'] == transition_type]
                    ax4.scatter(subset['pre_brexit_sp'], subset['post_brexit_sp'], 
                               c=transition_colors.get(transition_type, 'gray'), 
                               label=transition_type.replace('_', ' '), alpha=0.7, s=50)
                
                # Add diagonal line (no change)
                ax4.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, label='No Change')
                ax4.set_xlabel('Pre-Brexit Statistical Parity')
                ax4.set_ylabel('Post-Brexit Statistical Parity')
                ax4.set_title('Statistical Parity: Pre vs Post Brexit\nColor = Transition Type')
                ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.grid(True, alpha=0.3)
                ax4.set_xlim(-0.6, 0.6)
                ax4.set_ylim(-0.6, 0.6)
            
            plt.tight_layout()
            plt.savefig(self.config.output_dir / 'pattern_persistence_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create pattern persistence plot: {e}")
    
    def generate_report(self, analysis_results: Dict, insights: List[str]):
        """Generate comprehensive analysis report"""
        report_path = self.config.output_dir / f'{self.config.topic_filter.replace(" ", "_")}_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# 🎯 Deep Dive Analysis: {self.config.topic_filter}\n\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            for insight in insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # Pattern Transitions
            if 'pattern_transitions' in analysis_results:
                transitions = analysis_results['pattern_transitions']
                f.write("## 🔄 Pattern Transitions\n\n")
                
                if 'pattern_distribution' in transitions:
                    f.write("### Pattern Type Distribution\n")
                    for pattern_type, count in transitions['pattern_distribution'].items():
                        f.write(f"- **{pattern_type}**: {count} patterns\n")
                    f.write(f"\n**Net Change**: {transitions.get('net_change', 'N/A')} patterns\n\n")
                
                if 'country_impact' in transitions:
                    f.write("### Country-Level Impact\n")
                    f.write("| Country | Patterns Lost | Patterns Gained | Net Change |\n")
                    f.write("|---------|---------------|-----------------|------------|\n")
                    
                    for country, impact in transitions['country_impact'].items():
                        f.write(f"| {country} | {impact['lost_patterns']} | {impact['gained_patterns']} | {impact['net_change']:+d} |\n")
                    f.write("\n")
            
            # Cross-Attribute Analysis  
            if 'cross_attribute_patterns' in analysis_results:
                attr_analysis = analysis_results['cross_attribute_patterns']
                f.write("## 🎭 Cross-Attribute Analysis\n\n")
                f.write("| Attribute | Pre-Brexit Patterns | Post-Brexit Patterns | Net Change |\n")
                f.write("|-----------|--------------------|--------------------|------------|\n")
                
                for attribute, data in attr_analysis.items():
                    if isinstance(data, dict):
                        f.write(f"| {attribute} | {data['pre_patterns']} | {data['post_patterns']} | {data['net_change']:+d} |\n")
                f.write("\n")
            
            # Sub-topic Analysis
            if 'subtopic_analysis' in analysis_results:
                subtopic_analysis = analysis_results['subtopic_analysis']
                f.write("## 🔬 Sub-Topic Granular Analysis\n\n")
                
                for field_type, data in subtopic_analysis.items():
                    if isinstance(data, dict) and 'variations' in data:
                        f.write(f"### {field_type.title()}\n")
                        f.write("| Variation | Pre-Brexit Magnitude | Post-Brexit Magnitude | Change |\n")
                        f.write("|-----------|----------------------|----------------------|--------|\n")
                        
                        for var in data['variations']:
                            f.write(f"| {var['value'][:50]}... | {var['pre_magnitude']:.4f} | {var['post_magnitude']:.4f} | {var['magnitude_change']:+.4f} |\n")
                        f.write("\n")
            
            # A. INTERSECTIONAL GROUP DISADVANTAGE ANALYSIS
            if 'intersectional_disadvantage' in analysis_results:
                disadvantage = analysis_results['intersectional_disadvantage']
                f.write("## 🧠 A. Intersectional Group Disadvantage Analysis\n\n")
                f.write("*Identifying which groups face the strongest negative disparities and whether those disparities are statistically significant.*\n\n")
                
                if 'top_disadvantaged_groups' in disadvantage:
                    f.write("### Top 10 Most Disadvantaged Groups\n")
                    f.write("| Group | Attribute | Max Disadvantage | Pattern Type | Comparison |\n")
                    f.write("|-------|-----------|------------------|--------------|------------|\n")
                    
                    for group in disadvantage['top_disadvantaged_groups']:
                        f.write(f"| {group['group']} | {group['attribute']} | {group['max_disadvantage']:.4f} | {group['pattern_type']} | vs {group['comparison_group']} |\n")
                    f.write("\n")
                
                if 'intersectional_analysis' in disadvantage:
                    f.write("### Intersectional Groups (Multiple Attribute Disadvantages)\n")
                    for group, data in disadvantage['intersectional_analysis'].items():
                        f.write(f"**{group}**: Disadvantaged across {', '.join(data['attributes'])} (Total disadvantage: {data['total_disadvantage']:.4f})\n")
                    f.write("\n")
            
            # B. PATTERN PERSISTENCE/SHIFT ANALYSIS
            if 'pattern_persistence' in analysis_results:
                persistence = analysis_results['pattern_persistence']
                f.write("## 🔄 B. Pattern Persistence & Shift Analysis\n\n")
                f.write("*Understanding which group-level disparities are stable, emerged, or disappeared between pre- and post-Brexit models.*\n\n")
                
                if 'transition_summary' in persistence:
                    f.write("### Bias Pattern Transitions\n")
                    for transition_type, count in persistence['transition_summary'].items():
                        f.write(f"- **{transition_type.replace('_', ' ')}**: {count} patterns\n")
                    f.write("\n")
                
                if 'direction_summary' in persistence:
                    f.write("### Bias Direction Changes\n")
                    for direction, count in persistence['direction_summary'].items():
                        f.write(f"- **{direction.replace('_', ' ')}**: {count} patterns\n")
                    f.write("\n")
                
                if 'dramatic_changes' in persistence:
                    f.write("### Most Dramatic Changes (Top 5)\n")
                    f.write("| Comparison | Attribute | Pre-Brexit SP | Post-Brexit SP | Change | Transition Type |\n")
                    f.write("|------------|-----------|---------------|----------------|--------|----------------|\n")
                    
                    for change in persistence['dramatic_changes'][:5]:
                        f.write(f"| {change['comparison']} | {change['attribute']} | {change['pre_brexit_sp']:.4f} | {change['post_brexit_sp']:.4f} | {change['sp_change']:+.4f} | {change['transition_type']} |\n")
                    f.write("\n")
            
            # C. QUALITATIVE VIGNETTE COMPLETION ANALYSIS
            if 'qualitative_analysis' in analysis_results:
                qualitative = analysis_results['qualitative_analysis']
                f.write("## 📝 C. Qualitative Vignette Completion Analysis\n\n")
                f.write("*Moving from metrics to meaning - showing how bias manifests in text outputs.*\n\n")
                
                if 'max_divergence_cases' in qualitative:
                    f.write("### High-Priority Cases for Qualitative Analysis\n\n")
                    f.write("These cases show the largest statistical parity changes and warrant detailed examination of model completions:\n\n")
                    
                    for i, case in enumerate(qualitative['max_divergence_cases'], 1):
                        f.write(f"#### Case {i}: {case['group_pair']} ({case['attribute']})\n")
                        f.write(f"- **Topic**: {case['topic']}\n")
                        f.write(f"- **SP Change**: {case['sp_change']:+.4f}\n")
                        f.write(f"- **Analysis Focus**: {case['recommended_analysis']['focus_question']}\n")
                        f.write(f"- **Expected Bias Direction**: {case['recommended_analysis']['bias_direction']}\n")
                        f.write(f"- **Key Dimensions to Examine**: {', '.join(case['recommended_analysis']['analysis_dimensions'])}\n\n")
                
                if 'analysis_framework' in qualitative:
                    framework = qualitative['analysis_framework']
                    f.write("### Structured Analysis Framework\n\n")
                    f.write("For each high-priority case, examine vignette completions along these dimensions:\n\n")
                    f.write(f"1. **Tone Analysis**: {framework['tone_analysis']}\n")
                    f.write(f"2. **Legal Reasoning**: {framework['legal_reasoning']}\n") 
                    f.write(f"3. **Moral Framing**: {framework['moral_framing']}\n")
                    f.write(f"4. **Integration Assumptions**: {framework['integration_assumptions']}\n\n")
                    f.write(f"**Suggested Method**: {framework.get('suggested_method', 'Side-by-side completion comparison')}\n\n")
                    f.write(f"*Note*: {framework.get('note', 'Requires access to actual model completion texts')}\n\n")
            
            # Methodology
            f.write("## 📋 Methodology\n\n")
            f.write("- **Topic Filter**: Contains text matching '{self.config.topic_filter}'\n")
            f.write("- **Significance Testing**: FDR-corrected p-values\n")
            f.write("- **Pattern Classification**:\n")
            f.write("  - **NEWLY_EMERGED**: Not significant pre-Brexit, significant post-Brexit\n")
            f.write("  - **PERSISTENT**: Significant in both periods\n")
            f.write("  - **DISAPPEARED**: Significant pre-Brexit, not significant post-Brexit\n")
            f.write("- **Protected Attributes**: Country, Age, Religion, Gender\n\n")
            
            f.write("## 📁 Generated Files\n\n")
            f.write("- `pattern_transitions.png`: Pattern transition overview\n")
            f.write("- `cross_attribute_patterns.png`: Cross-attribute comparison\n")
            f.write("- `country_impact.png`: Country-level impact heatmap\n")
            f.write("- `magnitude_distribution.png`: Magnitude change distribution\n")
            f.write(f"- `{report_path.name}`: This report\n")
        
        return report_path
    
    def run_full_analysis(self):
        """Execute complete analysis pipeline"""
        print(f"🎯 Starting Deep Dive Analysis for: {self.config.topic_filter}")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Filter for topic
        print(f"🔍 Filtering data for topic: {self.config.topic_filter}")
        filtered_data = self.filter_topic_data()
        
        # Run analysis components
        analysis_results = {}
        
        print("📊 Analyzing pattern transitions...")
        analysis_results['pattern_transitions'] = self.analyze_pattern_transitions(filtered_data)
        
        print("🎭 Analyzing cross-attribute patterns...")
        analysis_results['cross_attribute_patterns'] = self.analyze_cross_attribute_patterns(filtered_data)
        
        print("🔬 Analyzing sub-topic granularity...")
        analysis_results['subtopic_analysis'] = self.analyze_subtopic_granularity(filtered_data)
        
        # NEW DEEP ANALYSES
        print("🧠 Analyzing intersectional group disadvantage...")
        analysis_results['intersectional_disadvantage'] = self.analyze_intersectional_group_disadvantage(filtered_data)
        
        print("🔄 Analyzing pattern persistence and shifts...")
        analysis_results['pattern_persistence'] = self.analyze_pattern_persistence_shifts(filtered_data)
        
        print("📝 Analyzing qualitative vignette targets...")
        analysis_results['qualitative_analysis'] = self.analyze_qualitative_vignette_completions(filtered_data)
        
        # Generate insights
        print("💡 Generating insights...")
        insights = self.generate_insights(analysis_results)
        
        # Create visualizations
        print("📈 Creating visualizations...")
        self.create_visualizations(filtered_data, analysis_results)
        
        # Generate report
        print("📝 Generating comprehensive report...")
        report_path = self.generate_report(analysis_results, insights)
        
        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"📄 Report: {report_path}")
        print("=" * 60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Topic Deep Dive Bias Analysis')
    parser.add_argument('--topic', required=True, help='Topic to analyze (e.g., "Intentions regarding work")')
    parser.add_argument('--base-dir', default='../..', help='Base directory containing fairness outputs')
    parser.add_argument('--output-dir', help='Output directory (default: topic_deep_dive/[topic_name])')
    parser.add_argument('--include-subtopics', action='store_true', default=True, help='Include sub-topic analysis')
    
    args = parser.parse_args()
    
    # Setup configuration
    base_data_dir = Path(args.base_dir).resolve() / "fairness" / "outputs"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        topic_name = args.topic.replace(" ", "_").replace("/", "_").lower()
        output_dir = Path(f"topic_deep_dive/{topic_name}")
    
    config = TopicAnalysisConfig(
        topic_filter=args.topic,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        include_subtopics=args.include_subtopics
    )
    
    # Run analysis
    analyzer = TopicDeepDiveAnalyzer(config)
    success = analyzer.run_full_analysis()
    
    if success:
        print(f"\n🎉 Analysis successful! Check {config.output_dir} for results.")
    else:
        print("\n❌ Analysis failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 