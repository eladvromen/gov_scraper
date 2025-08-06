#!/usr/bin/env python3
"""
📝 Vignette Completion Bias Analyzer
====================================

Deep dive into which specific vignette field completions drive bias patterns.
Links statistical bias patterns to actual vignette content variations.

Focus: Understanding which "work intentions" and "profession" combinations
drive the Syria/Nigeria/Myanmar bias changes.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class VignetteCompletionAnalyzer:
    """Analyzer for vignette-level bias patterns"""
    
    def __init__(self, focus_countries: List[str], focus_topic: str):
        self.focus_countries = focus_countries
        self.focus_topic = focus_topic
        self.data = {}
        self.vignette_structure = {}
        self.completion_bias_map = {}
        
    def load_vignette_data(self):
        """Load vignette structure and grant rate data"""
        print("📝 Loading vignette structure and completion data...")
        
        try:
            # Load complete vignette structure
            vignette_path = Path("/data/shil6369/vignettes/complete_vignettes.json")
            with open(vignette_path, 'r') as f:
                self.vignette_structure = json.load(f)
            
            print(f"✅ Loaded vignette structure with {len(self.vignette_structure)} fields")
            
            # Load grant rate analysis by field
            grant_rate_path = Path("/data/shil6369/gov_scraper/vignettes_analysis/comparative_fairness/normative/outputs/grant_rate_analysis/grant_rate_analysis_by_vignette_fields_enhanced_FIXED.csv")
            self.data['grant_rates'] = pd.read_csv(grant_rate_path)
            
            print(f"✅ Loaded {len(self.data['grant_rates'])} grant rate field analyses")
            
            # Load fairness comparison data
            fairness_path = Path("../../outputs/vector_drift/deduplicated_fairness_comparisons.csv")
            self.data['fairness_comparisons'] = pd.read_csv(fairness_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading vignette data: {e}")
            return False
    
    def extract_work_field_structure(self) -> Dict:
        """Extract work-related field structure from vignettes"""
        print("🔍 Extracting work-related field structure...")
        
        work_fields = {}
        
        # Look for work-related vignettes and their fields
        for vignette in self.vignette_structure:
            topic = vignette.get('topic', '')
            if 'work' in topic.lower() or any(work_keyword in topic.lower() for work_keyword in ['employment', 'job', 'career']):
                # Extract ordinal fields
                if 'ordinal_fields' in vignette:
                    for field_name, field_values in vignette['ordinal_fields'].items():
                        if any(work_keyword in field_name.lower() for work_keyword in ['work', 'profession', 'employment', 'job', 'career']):
                            work_fields[field_name] = field_values
                            print(f"📌 Found work field: {field_name} with {len(field_values)} values")
        
        # Also look for the specific work intentions vignette
        for vignette in self.vignette_structure:
            if vignette.get('topic') == 'Intentions regarding work in the UK':
                if 'ordinal_fields' in vignette:
                    for field_name, field_values in vignette['ordinal_fields'].items():
                        work_fields[field_name] = field_values
                        print(f"📌 Found work field: {field_name} with {len(field_values)} values")
                        print(f"   Values: {list(field_values.keys())}")
        
        return work_fields
    
    def analyze_completion_level_bias(self) -> Dict[str, Dict]:
        """Analyze bias at the individual completion level"""
        print("🧠 Analyzing bias at vignette completion level...")
        
        # Filter grant rate data for work-related topic
        work_grant_data = self.data['grant_rates'][
            self.data['grant_rates']['topic'].str.contains(self.focus_topic, case=False, na=False)
        ].copy()
        
        print(f"📊 Found {len(work_grant_data)} work-related field completions")
        
        completion_analysis = {
            'field_level_bias': {},
            'high_bias_completions': [],
            'country_specific_effects': {},
            'completion_bias_matrix': {}
        }
        
        # Group by field type (work intentions, profession, etc.)
        for field_name in work_grant_data['field_name'].unique():
            field_data = work_grant_data[work_grant_data['field_name'] == field_name]
            
            completion_analysis['field_level_bias'][field_name] = {
                'field_type': field_data['field_type'].iloc[0] if len(field_data) > 0 else 'unknown',
                'completions': [],
                'bias_range': 0,
                'most_biased_completion': None
            }
            
            max_bias = 0
            most_biased = None
            
            for _, completion in field_data.iterrows():
                completion_info = {
                    'field_value': completion['field_value'],
                    'pre_brexit_rate': completion['pre_brexit_raw_rate'],
                    'post_brexit_rate': completion['post_brexit_raw_rate'],
                    'rate_change': completion['cross_model_difference'],
                    'pre_normalized': completion['pre_brexit_normalized'],
                    'post_normalized': completion['post_brexit_normalized'],
                    'statistical_significance': completion['statistical_significance'],
                    'p_value': completion['p_value'],
                    'favors_model': completion['favors_model'],
                    'sample_size_pre': completion['pre_brexit_sample_size'],
                    'sample_size_post': completion['post_brexit_sample_size']
                }
                
                completion_analysis['field_level_bias'][field_name]['completions'].append(completion_info)
                
                # Track highest bias
                bias_magnitude = abs(completion['cross_model_difference'])
                if bias_magnitude > max_bias:
                    max_bias = bias_magnitude
                    most_biased = completion_info
                
                # Track high bias completions
                if completion['statistical_significance'] and bias_magnitude > 0.1:
                    completion_analysis['high_bias_completions'].append({
                        'field_name': field_name,
                        'completion': completion_info,
                        'bias_magnitude': bias_magnitude
                    })
            
            completion_analysis['field_level_bias'][field_name]['bias_range'] = max_bias
            completion_analysis['field_level_bias'][field_name]['most_biased_completion'] = most_biased
        
        return completion_analysis
    
    def map_completions_to_country_patterns(self, completion_analysis: Dict) -> Dict:
        """Map vignette completions to specific country bias patterns"""
        print("🗺️ Mapping completions to country bias patterns...")
        
        # This requires connecting grant rate field effects to country-specific patterns
        # We'll analyze which completions correlate with the country patterns we found
        
        country_completion_map = {}
        
        for country in self.focus_countries:
            country_completion_map[country] = {
                'advantageous_completions': [],
                'disadvantageous_completions': [],
                'neutral_completions': []
            }
            
            # Find completions that might affect this country specifically
            for field_name, field_info in completion_analysis['field_level_bias'].items():
                for completion in field_info['completions']:
                    field_value = completion['field_value'].lower()
                    
                    # Check if completion text references the country or related concepts
                    country_referenced = self._completion_references_country(field_value, country)
                    
                    if country_referenced or completion['statistical_significance']:
                        bias_effect = self._classify_completion_bias_effect(completion)
                        
                        completion_detail = {
                            'field_name': field_name,
                            'field_value': completion['field_value'],
                            'rate_change': completion['rate_change'],
                            'favors_model': completion['favors_model'],
                            'bias_magnitude': abs(completion['rate_change']),
                            'country_relevance': 'direct' if country_referenced else 'indirect'
                        }
                        
                        if bias_effect == 'positive':
                            country_completion_map[country]['advantageous_completions'].append(completion_detail)
                        elif bias_effect == 'negative':
                            country_completion_map[country]['disadvantageous_completions'].append(completion_detail)
                        else:
                            country_completion_map[country]['neutral_completions'].append(completion_detail)
        
        return country_completion_map
    
    def _completion_references_country(self, field_value: str, country: str) -> bool:
        """Check if a completion might reference a specific country context"""
        # This is heuristic - in practice, you'd need more sophisticated mapping
        country_keywords = {
            'syria': ['conflict', 'war', 'refugee', 'middle east', 'persecution'],
            'nigeria': ['economic', 'religious', 'ethnic', 'violence', 'lagos', 'abuja'],
            'myanmar': ['military', 'junta', 'rohingya', 'ethnic', 'burma', 'buddhist']
        }
        
        keywords = country_keywords.get(country.lower(), [])
        return any(keyword in field_value for keyword in keywords)
    
    def _classify_completion_bias_effect(self, completion: Dict) -> str:
        """Classify if completion has positive, negative, or neutral bias effect"""
        rate_change = completion['rate_change']
        favors_model = completion['favors_model']
        
        if not completion['statistical_significance']:
            return 'neutral'
        elif rate_change > 0.05:  # Substantial positive change
            return 'positive' if favors_model == 'post_brexit' else 'negative'
        elif rate_change < -0.05:  # Substantial negative change
            return 'negative' if favors_model == 'post_brexit' else 'positive'
        else:
            return 'neutral'
    
    def analyze_completion_interactions(self, completion_analysis: Dict) -> Dict:
        """Analyze interactions between different completion fields"""
        print("🔗 Analyzing completion field interactions...")
        
        interactions = {
            'work_intentions_vs_profession': {},
            'high_bias_combinations': [],
            'completion_correlation_matrix': {}
        }
        
        # Extract work intentions and profession data
        work_intentions_data = None
        profession_data = None
        
        for field_name, field_info in completion_analysis['field_level_bias'].items():
            if 'work' in field_name.lower() and 'intention' in field_name.lower():
                work_intentions_data = field_info
            elif 'profession' in field_name.lower():
                profession_data = field_info
        
        # Analyze interactions if both fields exist
        if work_intentions_data and profession_data:
            interactions['work_intentions_vs_profession'] = self._analyze_field_interaction(
                work_intentions_data, profession_data
            )
        
        # Find high bias combinations
        for field_name, field_info in completion_analysis['field_level_bias'].items():
            for completion in field_info['completions']:
                if completion['statistical_significance'] and abs(completion['rate_change']) > 0.15:
                    interactions['high_bias_combinations'].append({
                        'field_name': field_name,
                        'completion': completion['field_value'],
                        'bias_magnitude': abs(completion['rate_change']),
                        'direction': 'favors_post' if completion['rate_change'] > 0 else 'favors_pre',
                        'sample_size': completion['sample_size_pre'] + completion['sample_size_post']
                    })
        
        return interactions
    
    def _analyze_field_interaction(self, field1_data: Dict, field2_data: Dict) -> Dict:
        """Analyze interaction between two fields"""
        interaction_analysis = {
            'field1_name': 'work_intentions',
            'field2_name': 'profession',
            'bias_correlation': 0,
            'complementary_effects': [],
            'conflicting_effects': []
        }
        
        # Compare bias effects between fields
        field1_biases = [comp['rate_change'] for comp in field1_data['completions']]
        field2_biases = [comp['rate_change'] for comp in field2_data['completions']]
        
        if field1_biases and field2_biases:
            # Simple correlation (in practice, would need more sophisticated analysis)
            min_len = min(len(field1_biases), len(field2_biases))
            if min_len > 1:
                correlation = np.corrcoef(field1_biases[:min_len], field2_biases[:min_len])[0, 1]
                interaction_analysis['bias_correlation'] = correlation if not np.isnan(correlation) else 0
        
        return interaction_analysis
    
    def generate_completion_bias_report(self, completion_analysis: Dict, 
                                      country_mapping: Dict, interactions: Dict, 
                                      output_dir: Path) -> Path:
        """Generate detailed completion bias report"""
        print("📄 Generating vignette completion bias report...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"vignette_completion_bias_analysis.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# 📝 Vignette Completion Bias Analysis\n\n")
            f.write(f"**Focus Countries**: {', '.join(self.focus_countries)}\n")
            f.write(f"**Topic**: {self.focus_topic}\n")
            f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## 🎯 Executive Summary\n\n")
            total_high_bias = len(completion_analysis['high_bias_completions'])
            f.write(f"- **High Bias Completions Found**: {total_high_bias}\n")
            f.write(f"- **Fields Analyzed**: {len(completion_analysis['field_level_bias'])}\n")
            f.write(f"- **Countries Mapped**: {len(country_mapping)}\n\n")
            
            # Field-Level Bias Analysis
            f.write("## 📊 Field-Level Bias Analysis\n\n")
            f.write("*Detailed breakdown of bias by vignette field*\n\n")
            
            for field_name, field_info in completion_analysis['field_level_bias'].items():
                f.write(f"### {field_name}\n\n")
                f.write(f"- **Field Type**: {field_info['field_type']}\n")
                f.write(f"- **Number of Completions**: {len(field_info['completions'])}\n")
                f.write(f"- **Bias Range**: {field_info['bias_range']:.4f}\n\n")
                
                if field_info['most_biased_completion']:
                    mbc = field_info['most_biased_completion']
                    f.write(f"**Most Biased Completion**: {mbc['field_value']}\n")
                    f.write(f"- Pre-Brexit Rate: {mbc['pre_brexit_rate']:.3f}\n")
                    f.write(f"- Post-Brexit Rate: {mbc['post_brexit_rate']:.3f}\n")
                    f.write(f"- Rate Change: {mbc['rate_change']:+.3f}\n")
                    f.write(f"- Favors: {mbc['favors_model']}\n\n")
                
                # Table of all completions
                f.write("#### All Completions\n\n")
                f.write("| Completion | Pre-Rate | Post-Rate | Change | Significant | Favors |\n")
                f.write("|------------|----------|-----------|--------|-------------|--------|\n")
                
                for completion in sorted(field_info['completions'], 
                                       key=lambda x: abs(x['rate_change']), reverse=True):
                    f.write(f"| {completion['field_value'][:50]}... | "
                           f"{completion['pre_brexit_rate']:.3f} | "
                           f"{completion['post_brexit_rate']:.3f} | "
                           f"{completion['rate_change']:+.3f} | "
                           f"{'✓' if completion['statistical_significance'] else '✗'} | "
                           f"{completion['favors_model']} |\n")
                f.write("\n")
            
            # High Bias Completions
            f.write("## 🔥 High Bias Completions\n\n")
            f.write("*Completions with statistically significant bias > 0.1*\n\n")
            
            if completion_analysis['high_bias_completions']:
                f.write("| Field | Completion | Bias Magnitude | Direction | Sample Size |\n")
                f.write("|-------|------------|----------------|-----------|-------------|\n")
                
                for hbc in sorted(completion_analysis['high_bias_completions'], 
                                key=lambda x: x['bias_magnitude'], reverse=True):
                    completion = hbc['completion']
                    direction = 'Post-Brexit' if completion['rate_change'] > 0 else 'Pre-Brexit'
                    sample_size = completion['sample_size_pre'] + completion['sample_size_post']
                    
                    f.write(f"| {hbc['field_name']} | {completion['field_value'][:40]}... | "
                           f"{hbc['bias_magnitude']:.3f} | {direction} | {sample_size} |\n")
                f.write("\n")
            else:
                f.write("No high bias completions found with the current threshold.\n\n")
            
            # Country-Specific Effects
            f.write("## 🌍 Country-Specific Completion Effects\n\n")
            f.write("*How specific completions affect each focus country*\n\n")
            
            for country, mapping in country_mapping.items():
                f.write(f"### {country}\n\n")
                
                adv_count = len(mapping['advantageous_completions'])
                dis_count = len(mapping['disadvantageous_completions'])
                neu_count = len(mapping['neutral_completions'])
                
                f.write(f"- **Advantageous Completions**: {adv_count}\n")
                f.write(f"- **Disadvantageous Completions**: {dis_count}\n")
                f.write(f"- **Neutral Completions**: {neu_count}\n\n")
                
                # Show top advantageous completions
                if mapping['advantageous_completions']:
                    f.write("#### Top Advantageous Completions\n\n")
                    for comp in sorted(mapping['advantageous_completions'], 
                                     key=lambda x: x['bias_magnitude'], reverse=True)[:3]:
                        f.write(f"**{comp['field_name']}**: {comp['field_value']}\n")
                        f.write(f"- Rate Change: {comp['rate_change']:+.3f}\n")
                        f.write(f"- Relevance: {comp['country_relevance']}\n\n")
                
                # Show top disadvantageous completions
                if mapping['disadvantageous_completions']:
                    f.write("#### Top Disadvantageous Completions\n\n")
                    for comp in sorted(mapping['disadvantageous_completions'], 
                                     key=lambda x: x['bias_magnitude'], reverse=True)[:3]:
                        f.write(f"**{comp['field_name']}**: {comp['field_value']}\n")
                        f.write(f"- Rate Change: {comp['rate_change']:+.3f}\n")
                        f.write(f"- Relevance: {comp['country_relevance']}\n\n")
            
            # Interaction Analysis
            f.write("## 🔗 Completion Interaction Analysis\n\n")
            
            if interactions['high_bias_combinations']:
                f.write("### High Bias Completion Combinations\n\n")
                f.write("| Field | Completion | Bias | Direction | Sample Size |\n")
                f.write("|-------|------------|------|-----------|-------------|\n")
                
                for combo in sorted(interactions['high_bias_combinations'], 
                                  key=lambda x: x['bias_magnitude'], reverse=True):
                    f.write(f"| {combo['field_name']} | {combo['completion'][:40]}... | "
                           f"{combo['bias_magnitude']:.3f} | {combo['direction']} | "
                           f"{combo['sample_size']} |\n")
                f.write("\n")
            
            # Qualitative Analysis Framework
            f.write("## 🧠 Qualitative Analysis Framework\n\n")
            f.write("### Key Questions for Manual Completion Review\n\n")
            f.write("1. **Work Intentions Field**:\n")
            f.write("   - Which work intention completions favor/penalize specific countries?\n")
            f.write("   - How does 'actively pursue career advancement' vs 'focusing solely on safety' affect decisions?\n")
            f.write("   - Are certain countries assumed to have different work motivations?\n\n")
            
            f.write("2. **Profession Field**:\n")
            f.write("   - Do 'successful entrepreneur' vs 'unskilled laborer' completions show country bias?\n")
            f.write("   - Are certain countries' professional credentials treated differently?\n")
            f.write("   - How does 'recent graduate' completion interact with country of origin?\n\n")
            
            f.write("3. **Cross-Field Interactions**:\n")
            f.write("   - Do 'entrepreneur' + 'career advancement' combinations favor certain countries?\n")
            f.write("   - Are there problematic assumptions about country-profession relationships?\n")
            f.write("   - Which completion combinations drive the Myanmar/Syria/Nigeria pattern changes?\n\n")
            
            # Methodology
            f.write("## 📋 Methodology\n\n")
            f.write(f"- **Vignette Structure**: {len(self.vignette_structure)} fields analyzed\n")
            f.write(f"- **Grant Rate Data**: Field-level completion analysis\n")
            f.write(f"- **Bias Threshold**: |Rate Change| > 0.1 for high bias\n")
            f.write(f"- **Significance**: p < 0.05 statistical significance\n")
            f.write(f"- **Country Mapping**: Heuristic-based completion-country relevance\n\n")
        
        return report_path
    
    def create_completion_visualizations(self, completion_analysis: Dict, 
                                       country_mapping: Dict, output_dir: Path):
        """Create visualizations for completion-level bias"""
        print("📈 Creating completion bias visualizations...")
        
        try:
            # 1. Field-level bias heatmap
            self._plot_field_bias_heatmap(completion_analysis, output_dir)
            
            # 2. Completion bias distribution
            self._plot_completion_bias_distribution(completion_analysis, output_dir)
            
            # 3. Country-specific completion effects
            self._plot_country_completion_effects(country_mapping, output_dir)
            
        except Exception as e:
            print(f"Warning: Could not create some visualizations: {e}")
    
    def _plot_field_bias_heatmap(self, completion_analysis: Dict, output_dir: Path):
        """Plot heatmap of bias by field and completion"""
        field_names = []
        completion_names = []
        bias_values = []
        
        for field_name, field_info in completion_analysis['field_level_bias'].items():
            for completion in field_info['completions']:
                if completion['statistical_significance']:
                    field_names.append(field_name)
                    completion_names.append(completion['field_value'][:30] + "...")
                    bias_values.append(completion['rate_change'])
        
        if not bias_values:
            return
        
        # Create DataFrame for heatmap
        df = pd.DataFrame({
            'Field': field_names,
            'Completion': completion_names,
            'Bias': bias_values
        })
        
        # Pivot for heatmap
        pivot_df = df.pivot_table(values='Bias', index='Completion', columns='Field', fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Grant Rate Change (Post - Pre)'})
        plt.title('Vignette Completion Bias Heatmap\nWork-Related Fields')
        plt.tight_layout()
        plt.savefig(output_dir / 'completion_bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_completion_bias_distribution(self, completion_analysis: Dict, output_dir: Path):
        """Plot distribution of completion biases"""
        all_biases = []
        field_labels = []
        
        for field_name, field_info in completion_analysis['field_level_bias'].items():
            for completion in field_info['completions']:
                all_biases.append(completion['rate_change'])
                field_labels.append(field_name)
        
        if not all_biases:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of all biases
        ax1.hist(all_biases, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Bias')
        ax1.set_xlabel('Grant Rate Change')
        ax1.set_ylabel('Number of Completions')
        ax1.set_title('Distribution of Completion Biases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by field
        df = pd.DataFrame({'Bias': all_biases, 'Field': field_labels})
        unique_fields = df['Field'].unique()
        field_biases = [df[df['Field'] == field]['Bias'].values for field in unique_fields]
        
        bp = ax2.boxplot(field_biases, labels=[f[:15] + "..." for f in unique_fields], patch_artist=True)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Grant Rate Change')
        ax2.set_title('Completion Bias by Field')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'completion_bias_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_country_completion_effects(self, country_mapping: Dict, output_dir: Path):
        """Plot country-specific completion effects"""
        if not country_mapping:
            return
        
        countries = list(country_mapping.keys())
        adv_counts = [len(mapping['advantageous_completions']) for mapping in country_mapping.values()]
        dis_counts = [len(mapping['disadvantageous_completions']) for mapping in country_mapping.values()]
        
        x = np.arange(len(countries))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, adv_counts, width, label='Advantageous', alpha=0.8, color='green')
        bars2 = ax.bar(x + width/2, dis_counts, width, label='Disadvantageous', alpha=0.8, color='red')
        
        ax.set_xlabel('Countries')
        ax.set_ylabel('Number of Completions')
        ax.set_title('Country-Specific Completion Effects\nWork-Related Vignettes')
        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, str(int(height)), 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'country_completion_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_completion_analysis(self, output_dir: Path = None) -> bool:
        """Run complete vignette completion bias analysis"""
        if output_dir is None:
            output_dir = Path("vignette_completion_analysis")
        
        print(f"📝 Starting Vignette Completion Bias Analysis")
        print(f"Focus: {', '.join(self.focus_countries)} in '{self.focus_topic}'")
        print("=" * 60)
        
        # Load data
        if not self.load_vignette_data():
            return False
        
        # Extract work field structure
        work_fields = self.extract_work_field_structure()
        print(f"📊 Work fields found: {list(work_fields.keys())}")
        
        # Analyze completion-level bias
        completion_analysis = self.analyze_completion_level_bias()
        
        # Map to country patterns
        country_mapping = self.map_completions_to_country_patterns(completion_analysis)
        
        # Analyze interactions
        interactions = self.analyze_completion_interactions(completion_analysis)
        
        # Create visualizations
        self.create_completion_visualizations(completion_analysis, country_mapping, output_dir)
        
        # Generate report
        report_path = self.generate_completion_bias_report(
            completion_analysis, country_mapping, interactions, output_dir
        )
        
        print(f"\n✅ Vignette Completion Analysis Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Report: {report_path}")
        print("=" * 60)
        
        return True

def main():
    """Main execution function"""
    
    # Configuration
    focus_countries = ["Syria", "Nigeria", "Myanmar"]
    focus_topic = "Intentions regarding work in the UK"
    
    # Initialize analyzer
    analyzer = VignetteCompletionAnalyzer(focus_countries, focus_topic)
    
    # Run analysis
    success = analyzer.run_completion_analysis()
    
    if success:
        print("\n🎯 Key Deliverables Generated:")
        print("• Field-level bias analysis for work intentions and profession")
        print("• Specific completion combinations driving bias patterns")
        print("• Country-specific completion effects mapping")
        print("• High bias completion identification")
        print("• Interaction analysis between completion fields")
        print("• Comprehensive visualizations")
    else:
        print("\n❌ Analysis failed. Check data paths and requirements.")

if __name__ == "__main__":
    main() 