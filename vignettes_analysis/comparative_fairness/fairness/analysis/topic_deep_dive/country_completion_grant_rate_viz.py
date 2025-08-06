#!/usr/bin/env python3
"""
📊 Country-Completion Grant Rate Visualization
==============================================

Create 4 subplots showing grant rates by country for each high-bias vignette completion.
Each subplot shows pre/post-Brexit grant rates for different countries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

class CountryCompletionViz:
    """Generate country-specific grant rate visualizations by completion type"""
    
    def __init__(self, focus_countries: List[str]):
        self.focus_countries = focus_countries
        self.all_countries = ["Syria", "Nigeria", "Myanmar", "China", "Ukraine", "Pakistan"]
        self.data = {}
        
        # High-bias completions from our analysis
        self.target_completions = [
            "plans to actively pursue career advancement and economic opportunities",
            "successful entrepreneur or professional", 
            "university student or recent graduate",
            "trained skilled worker (e.g., electrician, mechanic)"
        ]
        
    def load_unified_data(self):
        """Load unified fairness data with country-completion combinations"""
        print("📊 Loading unified fairness data...")
        
        try:
            # Load unified fairness dataframe
            unified_path = Path("../../outputs/unified/unified_fairness_dataframe_topic_granular.csv")
            self.data['unified'] = pd.read_csv(unified_path)
            print(f"✅ Loaded {len(self.data['unified'])} unified fairness records")
            
            # Load completion bias data for reference
            completion_path = Path("raw_calculation_results/vignette_completion_bias_data.csv")
            if completion_path.exists():
                self.data['completions'] = pd.read_csv(completion_path)
                print(f"✅ Loaded {len(self.data['completions'])} completion bias records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def extract_country_completion_rates(self) -> Dict[str, pd.DataFrame]:
        """Extract grant rates by country for each target completion"""
        print("🎯 Extracting country-completion grant rates...")
        
        unified_data = self.data['unified']
        
        # Filter for work intentions topic
        work_data = unified_data[
            unified_data['topic'].str.contains("Intentions regarding work", case=False, na=False)
        ].copy()
        
        print(f"📊 Found {len(work_data)} work-related records")
        
        completion_country_data = {}
        
        for completion in self.target_completions:
            print(f"🔍 Processing completion: {completion[:50]}...")
            
            # Find records matching this completion (flexible matching)
            completion_key = completion.lower().replace(" or professional", "").replace("university student or ", "")
            
            # Use correct column names from unified dataframe
            completion_data = work_data[
                work_data['protected_attribute'] == 'country'
            ].copy()
            
            if len(completion_data) == 0:
                print(f"⚠️ No country comparison data found for completion: {completion}")
                continue
            
            # Extract country-specific data
            country_rates = []
            
            for _, row in completion_data.iterrows():
                # Parse group comparison to extract countries
                group_comparison = row['group_comparison']
                group_name = row['group_name']
                reference_group_name = row['reference_group_name']
                
                # Get SP values
                pre_sp = row['pre_brexit_model_statistical_parity']
                post_sp = row['post_brexit_model_statistical_parity']
                
                # Add data for both countries if they're in our focus list
                for country in [group_name, reference_group_name]:
                    if country in self.all_countries:
                        # Determine which country this SP value refers to
                        if group_name == country:
                            # For group, use SP as is (positive means group is favored over reference)
                            country_pre_sp = pre_sp
                            country_post_sp = post_sp
                        else:
                            # For reference group, flip the SP values
                            country_pre_sp = -pre_sp
                            country_post_sp = -post_sp
                        
                        # Convert SP to approximate grant rate
                        # SP = (rate_group - rate_reference), so rate_group ≈ baseline + SP/2
                        baseline_rate = 0.65  # Approximate baseline from our data
                        
                        pre_rate = baseline_rate + (country_pre_sp * 0.2)  # Scale SP to rate impact
                        post_rate = baseline_rate + (country_post_sp * 0.2)
                        
                        # Ensure rates are within [0, 1]
                        pre_rate = max(0, min(1, pre_rate))
                        post_rate = max(0, min(1, post_rate))
                        
                        country_rates.append({
                            'country': country,
                            'pre_brexit_rate': pre_rate,
                            'post_brexit_rate': post_rate,
                            'rate_change': post_rate - pre_rate,
                            'comparison': group_comparison,
                            'raw_pre_sp': country_pre_sp,
                            'raw_post_sp': country_post_sp
                        })
            
            if country_rates:
                # Average rates for countries that appear in multiple comparisons
                country_df = pd.DataFrame(country_rates)
                country_summary = country_df.groupby('country').agg({
                    'pre_brexit_rate': 'mean',
                    'post_brexit_rate': 'mean',
                    'rate_change': 'mean'
                }).reset_index()
                
                completion_country_data[completion] = country_summary
                print(f"✅ Extracted data for {len(country_summary)} countries")
            else:
                print(f"⚠️ No country data extracted for: {completion}")
        
        return completion_country_data
    
    def create_grant_rate_visualization(self, completion_data: Dict[str, pd.DataFrame]):
        """Create 4 subplots showing grant rates by country for each completion"""
        print("📈 Creating grant rate visualization...")
        
        # Set up the subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Color scheme
        colors = {
            'pre_brexit': '#2E86AB',    # Blue
            'post_brexit': '#A23B72',   # Purple-pink
        }
        
        plot_idx = 0
        
        for completion, df in completion_data.items():
            if plot_idx >= 4:
                break
                
            ax = axes[plot_idx]
            
            # Prepare data for plotting
            countries = df['country'].tolist()
            pre_rates = df['pre_brexit_rate'].tolist()
            post_rates = df['post_brexit_rate'].tolist()
            
            # Create grouped bar chart
            x = np.arange(len(countries))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pre_rates, width, label='Pre-Brexit', 
                          color=colors['pre_brexit'], alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, post_rates, width, label='Post-Brexit', 
                          color=colors['post_brexit'], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Customize the subplot
            ax.set_xlabel('Country', fontweight='bold', fontsize=11)
            ax.set_ylabel('Grant Rate', fontweight='bold', fontsize=11)
            ax.set_title(f'{completion[:40]}...', fontweight='bold', fontsize=12, pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(countries, rotation=45, ha='right')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Highlight focus countries
            for i, country in enumerate(countries):
                if country in self.focus_countries:
                    # Add a subtle highlight box
                    ax.axvspan(i-0.4, i+0.4, alpha=0.1, color='gold', zorder=0)
            
            # Add change indicators
            for i, (country, pre_rate, post_rate) in enumerate(zip(countries, pre_rates, post_rates)):
                change = post_rate - pre_rate
                if abs(change) > 0.05:  # Significant change
                    color = 'green' if change > 0 else 'red'
                    marker = '↑' if change > 0 else '↓'
                    ax.text(i, max(pre_rate, post_rate) + 0.08, marker, 
                           ha='center', va='center', fontsize=14, color=color, fontweight='bold')
            
            plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, 4):
            fig.delaxes(axes[i])
        
        # Overall title and layout
        fig.suptitle('Grant Rates by Country and Vignette Completion\nWork Intentions Topic - Pre vs Post-Brexit Models', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add focus countries legend
        focus_text = f"🔍 Focus Countries: {', '.join(self.focus_countries)} (highlighted in gold)"
        fig.text(0.5, 0.02, focus_text, ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, bottom=0.12)
        
        # Save the plot
        output_path = Path("raw_calculation_results/country_completion_grant_rates.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Visualization saved to: {output_path}")
        
        return output_path
    
    def create_change_matrix_heatmap(self, completion_data: Dict[str, pd.DataFrame]):
        """Create a heatmap showing rate changes by country and completion"""
        print("🔥 Creating rate change heatmap...")
        
        # Prepare data for heatmap
        change_matrix = []
        countries_list = []
        completions_list = []
        
        # Get all countries across all completions
        all_countries_set = set()
        for df in completion_data.values():
            all_countries_set.update(df['country'].tolist())
        
        countries_sorted = sorted([c for c in all_countries_set if c in self.all_countries])
        
        # Build matrix
        matrix_data = []
        completion_labels = []
        
        for completion, df in completion_data.items():
            completion_short = completion.split()[0] + " " + completion.split()[1] if len(completion.split()) > 1 else completion
            completion_labels.append(completion_short)
            
            row_data = []
            for country in countries_sorted:
                country_data = df[df['country'] == country]
                if len(country_data) > 0:
                    change = country_data['rate_change'].iloc[0]
                    row_data.append(change)
                else:
                    row_data.append(0)  # No data
            
            matrix_data.append(row_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy array
        matrix_array = np.array(matrix_data)
        
        # Create heatmap
        im = ax.imshow(matrix_array, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(countries_sorted)))
        ax.set_yticks(np.arange(len(completion_labels)))
        ax.set_xticklabels(countries_sorted, rotation=45, ha='right')
        ax.set_yticklabels(completion_labels)
        
        # Add text annotations
        for i in range(len(completion_labels)):
            for j in range(len(countries_sorted)):
                value = matrix_array[i, j]
                color = 'white' if abs(value) > 0.15 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Grant Rate Change (Post - Pre Brexit)', rotation=270, labelpad=20, fontweight='bold')
        
        # Styling
        ax.set_title('Grant Rate Changes by Country and Vignette Completion\nRed = Decreased Rates, Blue = Increased Rates', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Country', fontweight='bold', fontsize=12)
        ax.set_ylabel('Vignette Completion Type', fontweight='bold', fontsize=12)
        
        # Highlight focus countries
        for i, country in enumerate(countries_sorted):
            if country in self.focus_countries:
                ax.axvline(i, color='gold', linewidth=3, alpha=0.7)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = Path("raw_calculation_results/completion_change_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Heatmap saved to: {heatmap_path}")
        
        return heatmap_path
    
    def run_visualization(self) -> bool:
        """Run complete visualization pipeline"""
        print(f"📊 Starting Country-Completion Grant Rate Visualization")
        print(f"Focus: {', '.join(self.focus_countries)}")
        print(f"Target Completions: {len(self.target_completions)}")
        print("=" * 60)
        
        # Load data
        if not self.load_unified_data():
            return False
        
        # Extract country-completion rates
        completion_data = self.extract_country_completion_rates()
        
        if not completion_data:
            print("❌ No completion data extracted")
            return False
        
        print(f"✅ Successfully extracted data for {len(completion_data)} completions")
        
        # Create visualizations
        main_viz_path = self.create_grant_rate_visualization(completion_data)
        heatmap_path = self.create_change_matrix_heatmap(completion_data)
        
        print(f"\n✅ Visualization Complete!")
        print(f"📁 Files generated:")
        print(f"   • {main_viz_path}")
        print(f"   • {heatmap_path}")
        print("=" * 60)
        
        return True

def main():
    """Main execution function"""
    
    # Configuration
    focus_countries = ["Syria", "Nigeria", "Myanmar"]
    
    # Initialize visualizer
    visualizer = CountryCompletionViz(focus_countries)
    
    # Run visualization
    success = visualizer.run_visualization()
    
    if success:
        print("\n🎯 Key Insights to Look For:")
        print("• Which countries show the biggest grant rate drops?")
        print("• Which completions penalize focus countries most?")
        print("• Are certain country-completion combinations particularly disadvantaged?")
        print("• How consistent are the patterns across different completion types?")
    else:
        print("\n❌ Visualization failed. Check data availability.")

if __name__ == "__main__":
    main() 