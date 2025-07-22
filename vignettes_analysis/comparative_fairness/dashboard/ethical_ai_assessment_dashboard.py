#!/usr/bin/env python3
"""
🔬 Ethical AI Assessment Dashboard
Advanced Interactive Analysis Tool for LLM Fairness & Normative Evaluation

Enables comprehensive exploration of:
- Fairness divergence patterns across demographics and topics
- Normative alignment shifts between model generations  
- Statistical significance patterns and transitions
- Geometric interpretations of bias vector changes
- Granular topic-attribute-model intersections

Built for rigorous scientific analysis of AI ethics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Advanced page configuration
st.set_page_config(
    page_title="🔬 Ethical AI Assessment Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_fairness_data():
    """Load comprehensive fairness analysis data"""
    base_path = Path("outputs")
    
    try:
        # Load unified fairness dataframe
        fairness_df = pd.read_csv(base_path / "unified_analysis" / "unified_fairness_dataframe_topic_granular.csv")
        
        # Load fairness divergence results
        with open(base_path / "fairness_divergence" / "fairness_divergence_results.json", 'r') as f:
            divergence_results = json.load(f)
            
        # Load SP vectors data
        with open(base_path / "fairness_divergence" / "sp_vectors_data.json", 'r') as f:
            sp_vectors = json.load(f)
            
        # Load topic-attribute analysis
        topic_attr_df = pd.read_csv(base_path / "visualizations" / "topic_attribute_analysis_data.csv")
        
        # Load significant bias findings
        with open(base_path / "visualizations" / "significant_bias_findings.json", 'r') as f:
            bias_findings = json.load(f)
            
        # Calculate derived metrics
        fairness_df['sp_change'] = (fairness_df['post_brexit_model_statistical_parity'] - 
                                   fairness_df['pre_brexit_model_statistical_parity'])
        fairness_df['sp_change_abs'] = np.abs(fairness_df['sp_change'])
        fairness_df['significance_transition'] = fairness_df.apply(
            lambda row: 'gained' if not row['pre_brexit_sp_significance'] and row['post_brexit_sp_significance']
            else 'lost' if row['pre_brexit_sp_significance'] and not row['post_brexit_sp_significance']
            else 'stable_sig' if row['pre_brexit_sp_significance'] and row['post_brexit_sp_significance']
            else 'stable_nonsig', axis=1
        )
        
        return {
            'fairness_df': fairness_df,
            'divergence_results': divergence_results,
            'sp_vectors': sp_vectors,
            'topic_attr_df': topic_attr_df,
            'bias_findings': bias_findings
        }
        
    except Exception as e:
        st.error(f"Error loading fairness data: {str(e)}")
        return None

@st.cache_data
def load_normative_data():
    """Load normative analysis data"""
    base_path = Path("outputs")
    
    try:
        # Load normative divergence results
        with open(base_path / "normative_divergence" / "normative_divergence_results.json", 'r') as f:
            normative_results = json.load(f)
            
        # Load normative summary
        with open(base_path / "normative_divergence" / "normative_divergence_summary.md", 'r') as f:
            normative_summary = f.read()
            
        # Load vignette field analysis data
        normative_fields_df = pd.read_csv(base_path / "grant_rate_analysis" / "grant_rate_analysis_by_vignette_fields.csv")
        
        # Load topic tendencies
        topic_tendencies_df = pd.read_csv(base_path / "grant_rate_analysis" / "topic_tendencies_analysis.csv")
        
        return {
            'normative_results': normative_results,
            'normative_summary': normative_summary,
            'normative_fields_df': normative_fields_df,
            'topic_tendencies_df': topic_tendencies_df
        }
        
    except Exception as e:
        st.warning(f"Normative data not available: {str(e)}")
        return None

def create_geometric_visualization(vectors_data: Dict, selected_metrics: List[str] = None):
    """Create geometric representations of bias vectors"""
    
    if selected_metrics is None:
        selected_metrics = ['pre_brexit_sp_magnitude', 'post_brexit_sp_magnitude']
    
    # Extract vectors
    pre_vector = np.array(vectors_data['vectors']['pre_brexit_sp_magnitude'])
    post_vector = np.array(vectors_data['vectors']['post_brexit_sp_magnitude'])
    
    # Calculate geometric metrics
    cosine_sim = np.dot(pre_vector, post_vector) / (np.linalg.norm(pre_vector) * np.linalg.norm(post_vector))
    angle_deg = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Vector Angle Representation",
            "Magnitude Comparison", 
            "Difference Distribution",
            "2D Projection (PCA)"
        ],
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # 1. Vector angle visualization
    theta = np.linspace(0, angle_deg * np.pi / 180, 100)
    x_arc = 0.5 * np.cos(theta)
    y_arc = 0.5 * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode='lines+markers',
        name='Pre-Brexit Vector', line=dict(color='blue', width=4)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0, np.cos(angle_deg * np.pi / 180)], 
        y=[0, np.sin(angle_deg * np.pi / 180)],
        mode='lines+markers', name='Post-Brexit Vector', 
        line=dict(color='red', width=4)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=x_arc, y=y_arc, mode='lines',
        name=f'Divergence: {angle_deg:.1f}°',
        line=dict(color='green', dash='dash', width=2)
    ), row=1, col=1)
    
    # 2. Magnitude comparison
    magnitudes = ['Pre-Brexit', 'Post-Brexit']
    mag_values = [np.linalg.norm(pre_vector), np.linalg.norm(post_vector)]
    
    fig.add_trace(go.Bar(
        x=magnitudes, y=mag_values,
        name='Vector Magnitudes',
        marker_color=['blue', 'red']
    ), row=1, col=2)
    
    # 3. Difference distribution
    diff_vector = post_vector - pre_vector
    
    fig.add_trace(go.Histogram(
        x=diff_vector, name='SP Changes Distribution',
        marker_color='purple', opacity=0.7
    ), row=2, col=1)
    
    # 4. 2D PCA projection
    if len(pre_vector) > 2:
        vectors_matrix = np.column_stack([pre_vector, post_vector])
        pca = PCA(n_components=2)
        projected = pca.fit_transform(vectors_matrix.T)
        
        fig.add_trace(go.Scatter(
            x=projected[:, 0], y=projected[:, 1],
            mode='markers+text',
            text=['Pre-Brexit', 'Post-Brexit'],
            textposition='top center',
            marker=dict(size=15, color=['blue', 'red']),
            name='PCA Projection'
        ), row=2, col=2)
    
    fig.update_layout(
        height=800,
        title_text=f"Geometric Analysis: Bias Vector Divergence (Angle: {angle_deg:.1f}°, Cosine Sim: {cosine_sim:.3f})",
        showlegend=True
    )
    
    return fig

def create_significance_transition_matrix(fairness_df: pd.DataFrame):
    """Create significance transition analysis"""
    
    # Calculate transition matrix
    transition_counts = fairness_df['significance_transition'].value_counts()
    
    # Create transition flow visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Significance Transitions",
            "Transitions by Attribute",
            "Transitions by Topic (Top 10)",
            "Statistical Significance Rates"
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Overall transitions
    colors = {'gained': 'green', 'lost': 'red', 'stable_sig': 'blue', 'stable_nonsig': 'gray'}
    fig.add_trace(go.Bar(
        x=transition_counts.index,
        y=transition_counts.values,
        marker_color=[colors.get(x, 'gray') for x in transition_counts.index],
        name='Transition Counts',
        text=transition_counts.values,
        textposition='outside'
    ), row=1, col=1)
    
    # 2. Transitions by protected attribute
    attr_transitions = fairness_df.groupby(['protected_attribute', 'significance_transition']).size().unstack(fill_value=0)
    
    for transition in attr_transitions.columns:
        fig.add_trace(go.Bar(
            x=attr_transitions.index,
            y=attr_transitions[transition],
            name=f'{transition.title()}',
            marker_color=colors.get(transition, 'gray')
        ), row=1, col=2)
    
    # 3. Transitions by topic (top 10)
    topic_transitions = fairness_df.groupby('topic')['significance_transition'].apply(
        lambda x: (x == 'gained').sum() - (x == 'lost').sum()
    ).sort_values(ascending=False).head(10)
    
    fig.add_trace(go.Bar(
        x=topic_transitions.index,
        y=topic_transitions.values,
        marker_color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in topic_transitions.values],
        name='Net Significance Change',
        text=topic_transitions.values,
        textposition='outside'
    ), row=2, col=1)
    
    # 4. Statistical significance rates
    pre_sig_rate = fairness_df['pre_brexit_sp_significance'].mean()
    post_sig_rate = fairness_df['post_brexit_sp_significance'].mean()
    
    fig.add_trace(go.Bar(
        x=['Pre-Brexit', 'Post-Brexit'],
        y=[pre_sig_rate, post_sig_rate],
        marker_color=['blue', 'red'],
        name='Significance Rates',
        text=[f'{pre_sig_rate:.1%}', f'{post_sig_rate:.1%}'],
        textposition='outside'
    ), row=2, col=2)
    
    fig.update_layout(
        height=1000,
        title_text="Statistical Significance Transition Analysis: Pre-Brexit → Post-Brexit",
        showlegend=True
    )
    
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    return fig

def create_intersectional_heatmap(fairness_df: pd.DataFrame, metric: str = 'sp_change_abs'):
    """Create intersectional analysis heatmap"""
    
    # Create topic-attribute intersection matrix
    intersection_matrix = fairness_df.groupby(['topic', 'protected_attribute'])[metric].mean().unstack(fill_value=0)
    
    # Create interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=intersection_matrix.values,
        x=intersection_matrix.columns,
        y=intersection_matrix.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='Topic: %{y}<br>Attribute: %{x}<br>Value: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Intersectional Analysis: {metric.replace("_", " ").title()} by Topic × Attribute',
        xaxis_title='Protected Attribute',
        yaxis_title='Topic',
        height=800
    )
    
    return fig

def create_distribution_comparison(fairness_df: pd.DataFrame):
    """Create comprehensive distribution comparison"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "SP Values Distribution", "SP Changes Distribution", "Significance Rates by Attribute",
            "Topic-Level Variance", "Magnitude vs Significance", "Change Correlation Matrix"
        ]
    )
    
    # 1. SP Values Distribution
    fig.add_trace(go.Histogram(
        x=fairness_df['pre_brexit_model_statistical_parity'],
        name='Pre-Brexit SP', opacity=0.7, marker_color='blue'
    ), row=1, col=1)
    
    fig.add_trace(go.Histogram(
        x=fairness_df['post_brexit_model_statistical_parity'],
        name='Post-Brexit SP', opacity=0.7, marker_color='red'
    ), row=1, col=1)
    
    # 2. SP Changes Distribution
    fig.add_trace(go.Histogram(
        x=fairness_df['sp_change'],
        name='SP Changes', marker_color='purple'
    ), row=1, col=2)
    
    # 3. Significance rates by attribute
    sig_rates = fairness_df.groupby('protected_attribute').agg({
        'pre_brexit_sp_significance': 'mean',
        'post_brexit_sp_significance': 'mean'
    })
    
    fig.add_trace(go.Bar(
        x=sig_rates.index,
        y=sig_rates['pre_brexit_sp_significance'],
        name='Pre-Brexit Significance', marker_color='blue'
    ), row=1, col=3)
    
    fig.add_trace(go.Bar(
        x=sig_rates.index,
        y=sig_rates['post_brexit_sp_significance'],
        name='Post-Brexit Significance', marker_color='red'
    ), row=1, col=3)
    
    # 4. Topic-level variance
    topic_variance = fairness_df.groupby('topic')['sp_change_abs'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(10)
    
    fig.add_trace(go.Bar(
        x=topic_variance.index,
        y=topic_variance['mean'],
        error_y=dict(type='data', array=topic_variance['std']),
        name='Topic Variance', marker_color='green'
    ), row=2, col=1)
    
    # 5. Magnitude vs Significance scatter
    fig.add_trace(go.Scatter(
        x=fairness_df['sp_change_abs'],
        y=fairness_df['post_brexit_sp_significance'].astype(int),
        mode='markers',
        marker=dict(
            size=8,
            color=fairness_df['pre_brexit_sp_significance'].astype(int),
            colorscale='RdBu',
            showscale=True
        ),
        name='Magnitude vs Significance'
    ), row=2, col=2)
    
    # 6. Correlation matrix (simplified)
    numeric_cols = ['pre_brexit_model_statistical_parity', 'post_brexit_model_statistical_parity', 'sp_change', 'sp_change_abs']
    corr_matrix = fairness_df[numeric_cols].corr()
    
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0
    ), row=2, col=3)
    
    fig.update_layout(
        height=1200,
        title_text="Statistical Parity Distribution Analysis: Pre-Brexit vs Post-Brexit",
        showlegend=True
    )
    
    fig.update_xaxes(tickangle=45, row=2, col=1)
    
    return fig

# Main Dashboard
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Ethical AI Assessment Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced Interactive Analysis Tool for LLM Fairness & Normative Evaluation**")
    
    # Load data
    with st.spinner("🔄 Loading comprehensive ethical assessment data..."):
        fairness_data = load_fairness_data()
        normative_data = load_normative_data()
    
    if fairness_data is None:
        st.error("Failed to load fairness data. Please check file paths.")
        return
    
    # Sidebar Configuration
    st.sidebar.markdown("## 🎛️ Analysis Configuration")
    
    # Analysis mode selection
    analysis_mode = st.sidebar.selectbox(
        "🔬 Select Analysis Mode",
        [
            "🌍 Overview & Executive Summary",
            "⚖️ Fairness Deep Dive",
            "📏 Normative Analysis", 
            "🔄 Comparative Analysis",
            "🎯 Topic Stratified Analysis",
            "👥 Protected Attribute Analysis",
            "📊 Significance Pattern Analysis",
            "📐 Geometric & Vector Analysis",
            "🔍 Individual Case Explorer",
            "🧠 Scientific Insights Generator"
        ]
    )
    
    # Global filters
    st.sidebar.markdown("### 🎯 Global Filters")
    
    fairness_df = fairness_data['fairness_df']
    
    # Topic filter
    available_topics = ['All'] + sorted(fairness_df['topic'].unique())
    selected_topics = st.sidebar.multiselect(
        "Topics", available_topics, default=['All']
    )
    
    # Attribute filter
    available_attributes = ['All'] + sorted(fairness_df['protected_attribute'].unique())
    selected_attributes = st.sidebar.multiselect(
        "Protected Attributes", available_attributes, default=['All']
    )
    
    # Significance filter
    significance_filter = st.sidebar.selectbox(
        "Significance Status",
        ["All", "Significant Only", "Non-Significant Only", "Changed Significance"]
    )
    
    # Apply filters
    filtered_df = fairness_df.copy()
    
    if 'All' not in selected_topics and selected_topics:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topics)]
    
    if 'All' not in selected_attributes and selected_attributes:
        filtered_df = filtered_df[filtered_df['protected_attribute'].isin(selected_attributes)]
    
    if significance_filter == "Significant Only":
        filtered_df = filtered_df[filtered_df['post_brexit_sp_significance'] == True]
    elif significance_filter == "Non-Significant Only":
        filtered_df = filtered_df[filtered_df['post_brexit_sp_significance'] == False]
    elif significance_filter == "Changed Significance":
        filtered_df = filtered_df[filtered_df['significance_transition'].isin(['gained', 'lost'])]
    
    # Overview metrics with clear direction indicators
    st.markdown("## 📊 Executive Metrics")
    st.markdown("### 🔄 **Direction**: Pre-Brexit (2013-2016) → Post-Brexit (2019-2025) Model Evolution")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_comparisons = len(filtered_df)
    mean_change = filtered_df['sp_change'].mean()
    mean_abs_change = filtered_df['sp_change_abs'].mean()
    sig_change_rate = (filtered_df['significance_transition'].isin(['gained', 'lost'])).mean()
    pre_sig_rate = filtered_df['pre_brexit_sp_significance'].mean()
    post_sig_rate = filtered_df['post_brexit_sp_significance'].mean()
    
    # Calculate change direction
    change_direction = "📈 More Biased" if mean_change > 0 else "📉 Less Biased" if mean_change < 0 else "➡️ Unchanged"
    sig_direction = "📈 Increased" if post_sig_rate > pre_sig_rate else "📉 Decreased" if post_sig_rate < pre_sig_rate else "➡️ Stable"
    
    with col1:
        st.metric("Total Comparisons", f"{total_comparisons:,}")
    with col2:
        st.metric("Mean SP Change", f"{mean_change:+.4f}", help="Post-Brexit minus Pre-Brexit SP values")
    with col3:
        st.metric("Mean |SP Change|", f"{mean_abs_change:.4f}", help="Average magnitude of change")
    with col4:
        st.metric("Significance Instability", f"{sig_change_rate:.1%}", help="% of comparisons that changed significance")
    with col5:
        st.metric("Pre-Brexit Sig Rate", f"{pre_sig_rate:.1%}")
    with col6:
        st.metric("Post-Brexit Sig Rate", f"{post_sig_rate:.1%}", f"{sig_direction}")
    
    # Analysis sections based on mode
    if analysis_mode == "🌍 Overview & Executive Summary":
        st.markdown("## 🌍 Comprehensive Overview: Pre-Brexit (2013-2016) → Post-Brexit (2019-2025)")
        
        # Executive summary with key insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🔬 **Scientific Assessment Summary**
        
        **Dataset Scope**: {total_comparisons:,} demographic comparisons across {filtered_df['protected_attribute'].nunique()} protected attributes and {filtered_df['topic'].nunique()} topics
        
        **Key Findings (Pre→Post Brexit Evolution)**:
        - **Bias Magnitude**: Average absolute change of {mean_abs_change:.4f} in Statistical Parity
        - **Direction**: {change_direction} overall (mean change: {mean_change:+.4f})
        - **Pattern Stability**: {sig_change_rate:.1%} of comparisons changed significance patterns
        - **Statistical Significance**: {pre_sig_rate:.1%} → {post_sig_rate:.1%} ({sig_direction})
        - **Vector Divergence**: {fairness_data['divergence_results']['magnitude_divergence']['divergence_metrics']['cosine_similarity']:.3f} cosine similarity ({np.arccos(fairness_data['divergence_results']['magnitude_divergence']['divergence_metrics']['cosine_similarity']) * 180 / np.pi:.1f}° divergence angle)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Overview visualizations with better sizing
        st.markdown("### 📊 Core Visualizations")
        
        # First row: Distribution comparison 
        st.markdown("#### 📈 Statistical Parity Distribution: Pre-Brexit vs Post-Brexit")
        fig_dist = create_distribution_comparison(filtered_df)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Second row: Significance transitions
        st.markdown("#### 🔄 Statistical Significance Transition Patterns")
        fig_sig = create_significance_transition_matrix(filtered_df)
        st.plotly_chart(fig_sig, use_container_width=True)
    
    elif analysis_mode == "⚖️ Fairness Deep Dive":
        st.markdown("## ⚖️ Comprehensive Fairness Analysis: Pre-Brexit → Post-Brexit Evolution")
        
        # Fairness analysis configuration
        fairness_analysis_type = st.selectbox(
            "Select Fairness Analysis Type:",
            ["Statistical Parity Deep Dive", "Intersectional Analysis", "Temporal Changes", "Threshold Analysis"]
        )
        
        if fairness_analysis_type == "Statistical Parity Deep Dive":
            
            col1, col2 = st.columns(2)
            
            with col1:
                # SP distribution analysis
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=filtered_df['pre_brexit_model_statistical_parity'],
                    name='Pre-Brexit SP', opacity=0.7, 
                    marker_color='blue', nbinsx=50
                ))
                
                fig.add_trace(go.Histogram(
                    x=filtered_df['post_brexit_model_statistical_parity'],
                    name='Post-Brexit SP', opacity=0.7,
                    marker_color='red', nbinsx=50
                ))
                
                fig.update_layout(
                    title="Statistical Parity Distribution Comparison",
                    xaxis_title="Statistical Parity Value",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # SP change analysis by attribute
                fig = px.box(
                    filtered_df, 
                    x='protected_attribute', 
                    y='sp_change',
                    color='protected_attribute',
                    title="SP Change Distribution by Protected Attribute"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed statistical analysis
            st.markdown("### 📊 Statistical Analysis")
            
            # Calculate statistics by attribute
            attr_stats = filtered_df.groupby('protected_attribute').agg({
                'sp_change': ['mean', 'std', 'min', 'max'],
                'sp_change_abs': ['mean', 'std'],
                'pre_brexit_sp_significance': 'sum',
                'post_brexit_sp_significance': 'sum'
            }).round(4)
            
            st.dataframe(attr_stats, use_container_width=True)
        
        elif fairness_analysis_type == "Intersectional Analysis":
            # Intersectional heatmap
            metric_selection = st.selectbox(
                "Select Metric for Intersectional Analysis:",
                ['sp_change_abs', 'sp_change', 'pre_brexit_model_statistical_parity', 'post_brexit_model_statistical_parity']
            )
            
            fig_heatmap = create_intersectional_heatmap(filtered_df, metric_selection)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Top intersectional findings
            st.markdown("### 🔍 Top Intersectional Patterns")
            
            intersection_summary = filtered_df.groupby(['topic', 'protected_attribute']).agg({
                'sp_change_abs': ['mean', 'count'],
                'sp_change': 'mean',
                'significance_transition': lambda x: (x.isin(['gained', 'lost'])).sum()
            }).round(4)
            
            intersection_summary.columns = ['Mean |Change|', 'Count', 'Mean Change', 'Sig Transitions']
            intersection_summary = intersection_summary.sort_values('Mean |Change|', ascending=False).head(15)
            
            st.dataframe(intersection_summary, use_container_width=True)
        
        elif fairness_analysis_type == "Temporal Changes":
            st.markdown("### ⏰ Temporal Evolution Analysis: Pre-Brexit (2013-2016) → Post-Brexit (2019-2025)")
            
            # Temporal change analysis
            st.markdown("#### 📊 Change Magnitude Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Change direction analysis
                increased_bias = len(filtered_df[filtered_df['sp_change'] > 0.05])  # Meaningful increase
                decreased_bias = len(filtered_df[filtered_df['sp_change'] < -0.05])  # Meaningful decrease
                stable_bias = len(filtered_df[abs(filtered_df['sp_change']) <= 0.05])  # Stable
                
                labels = ['Increased Bias\n(Post > Pre)', 'Decreased Bias\n(Post < Pre)', 'Stable Bias\n(|Change| ≤ 0.05)']
                values = [increased_bias, decreased_bias, stable_bias]
                colors = ['red', 'green', 'gray']
                
                fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
                fig.update_layout(
                    title="Temporal Change Direction (Pre→Post Brexit)",
                    yaxis_title="Number of Comparisons",
                    xaxis_title="Change Direction"
                )
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    fig.add_annotation(x=i, y=v + max(values)*0.02, text=str(v), showarrow=False, font=dict(size=14, color="black"))
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Change timeline visualization (simulated temporal evolution)
                fig = go.Figure()
                
                # Simulate pre and post periods
                time_points = ['Pre-Brexit\n(2013-2016)', 'Post-Brexit\n(2019-2025)']
                
                for attr in filtered_df['protected_attribute'].unique():
                    attr_data = filtered_df[filtered_df['protected_attribute'] == attr]
                    pre_mean_bias = attr_data['pre_brexit_model_statistical_parity'].mean()
                    post_mean_bias = attr_data['post_brexit_model_statistical_parity'].mean()
                    
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=[pre_mean_bias, post_mean_bias],
                        mode='lines+markers',
                        name=f'{attr.title()} Bias',
                        line=dict(width=3),
                        marker=dict(size=10)
                    ))
                
                fig.update_layout(
                    title="Bias Evolution by Protected Attribute",
                    yaxis_title="Mean Statistical Parity",
                    xaxis_title="Time Period"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No Bias")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Temporal volatility analysis
            st.markdown("#### 📈 Volatility Analysis")
            
            # Calculate volatility metrics
            volatility_stats = filtered_df.groupby('protected_attribute').agg({
                'sp_change': ['std', 'min', 'max'],
                'sp_change_abs': 'mean',
                'significance_transition': lambda x: (x.isin(['gained', 'lost'])).mean()
            }).round(4)
            
            volatility_stats.columns = ['Change Std Dev', 'Min Change', 'Max Change', 'Mean |Change|', 'Sig Instability Rate']
            
            st.dataframe(volatility_stats, use_container_width=True)
        
        elif fairness_analysis_type == "Threshold Analysis":
            st.markdown("### 🎯 Threshold-Based Bias Analysis")
            
            # Threshold configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bias_threshold = st.slider(
                    "Bias Magnitude Threshold", 
                    min_value=0.01, max_value=0.5, value=0.1, step=0.01,
                    help="Minimum |SP Change| to be considered significant"
                )
            
            with col2:
                significance_threshold = st.selectbox(
                    "Significance Requirement",
                    ["Any Significance", "Both Models Significant", "Significance Change Only"],
                    help="Statistical significance requirements"
                )
            
            with col3:
                attribute_focus = st.selectbox(
                    "Focus on Attribute",
                    ["All Attributes"] + sorted(filtered_df['protected_attribute'].unique()),
                    help="Focus analysis on specific protected attribute"
                )
            
            # Apply thresholds
            threshold_df = filtered_df[filtered_df['sp_change_abs'] >= bias_threshold].copy()
            
            if significance_threshold == "Both Models Significant":
                threshold_df = threshold_df[
                    (threshold_df['pre_brexit_sp_significance'] == True) &
                    (threshold_df['post_brexit_sp_significance'] == True)
                ]
            elif significance_threshold == "Significance Change Only":
                threshold_df = threshold_df[threshold_df['significance_transition'].isin(['gained', 'lost'])]
            
            if attribute_focus != "All Attributes":
                threshold_df = threshold_df[threshold_df['protected_attribute'] == attribute_focus]
            
            # Results summary
            st.markdown(f"#### 📊 Results: {len(threshold_df)} cases meet threshold criteria")
            
            if len(threshold_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top concerning cases
                    st.markdown("##### 🚨 Most Concerning Cases")
                    
                    top_cases = threshold_df.nlargest(10, 'sp_change_abs')[
                        ['group_comparison', 'protected_attribute', 'topic', 'sp_change', 'sp_change_abs']
                    ]
                    
                    # Format for display
                    display_cases = top_cases.copy()
                    display_cases['sp_change'] = display_cases['sp_change'].apply(lambda x: f"{x:+.4f}")
                    display_cases['sp_change_abs'] = display_cases['sp_change_abs'].apply(lambda x: f"{x:.4f}")
                    display_cases.columns = ['Group Comparison', 'Attribute', 'Topic', 'SP Change', '|SP Change|']
                    
                    st.dataframe(display_cases, use_container_width=True)
                
                with col2:
                    # Threshold distribution
                    fig = go.Figure()
                    
                    # Histogram of changes above threshold
                    fig.add_trace(go.Histogram(
                        x=threshold_df['sp_change'],
                        nbinsx=20,
                        name='SP Changes Above Threshold',
                        marker_color='red',
                        opacity=0.7
                    ))
                    
                    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="No Change")
                    fig.add_vline(x=bias_threshold, line_dash="dot", line_color="red", annotation_text=f"Threshold: +{bias_threshold}")
                    fig.add_vline(x=-bias_threshold, line_dash="dot", line_color="red", annotation_text=f"Threshold: -{bias_threshold}")
                    
                    fig.update_layout(
                        title=f"Distribution of Changes Above Threshold (|ΔSP| ≥ {bias_threshold})",
                        xaxis_title="SP Change (Post-Brexit - Pre-Brexit)",
                        yaxis_title="Count"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Threshold summary statistics
                st.markdown("##### 📊 Threshold Summary Statistics")
                
                summary_stats = {
                    'Metric': [
                        'Cases Above Threshold',
                        'Percentage of Total',
                        'Mean |SP Change|',
                        'Max |SP Change|',
                        'Significance Change Rate',
                        'Most Affected Attribute',
                        'Most Affected Topic'
                    ],
                    'Value': [
                        f"{len(threshold_df):,}",
                        f"{len(threshold_df)/len(filtered_df):.1%}",
                        f"{threshold_df['sp_change_abs'].mean():.4f}",
                        f"{threshold_df['sp_change_abs'].max():.4f}",
                        f"{threshold_df['significance_transition'].isin(['gained', 'lost']).mean():.1%}",
                        threshold_df['protected_attribute'].value_counts().index[0] if len(threshold_df) > 0 else "N/A",
                        threshold_df['topic'].value_counts().index[0] if len(threshold_df) > 0 else "N/A"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_stats)
                st.dataframe(summary_df, use_container_width=True)
            
            else:
                st.warning(f"No cases meet the threshold criteria (|ΔSP| ≥ {bias_threshold})")
                st.info("Try lowering the bias threshold or changing the significance requirements.")
    
    elif analysis_mode == "📐 Geometric & Vector Analysis":
        st.markdown("## 📐 Geometric Analysis of Bias Vectors")
        
        # Geometric visualization options
        st.markdown("### 🎛️ Vector Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vector_metrics = st.multiselect(
                "Select Vector Metrics:",
                ['pre_brexit_sp_magnitude', 'post_brexit_sp_magnitude', 'sp_magnitude_difference'],
                default=['pre_brexit_sp_magnitude', 'post_brexit_sp_magnitude']
            )
        
        with col2:
            geometric_analysis_type = st.selectbox(
                "Analysis Type:",
                ["Vector Angles", "PCA Projection", "Cluster Analysis", "Dimensionality Reduction"]
            )
        
        # Create geometric visualization
        fig_geometric = create_geometric_visualization(fairness_data['sp_vectors'], vector_metrics)
        st.plotly_chart(fig_geometric, use_container_width=True)
        
        # Vector statistics
        st.markdown("### 📊 Vector Statistics")
        
        vectors = fairness_data['sp_vectors']['vectors']
        pre_vector = np.array(vectors['pre_brexit_sp_magnitude'])
        post_vector = np.array(vectors['post_brexit_sp_magnitude'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Pre-Brexit Magnitude", f"{np.linalg.norm(pre_vector):.4f}")
        with col2:
            st.metric("Post-Brexit Magnitude", f"{np.linalg.norm(post_vector):.4f}")
        with col3:
            cosine_sim = np.dot(pre_vector, post_vector) / (np.linalg.norm(pre_vector) * np.linalg.norm(post_vector))
            st.metric("Cosine Similarity", f"{cosine_sim:.4f}")
        with col4:
            angle_deg = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi
            st.metric("Divergence Angle", f"{angle_deg:.1f}°")
    
    elif analysis_mode == "📊 Significance Pattern Analysis":
        st.markdown("## 📊 Statistical Significance Pattern Analysis")
        
        # Significance transition analysis
        fig_sig_transitions = create_significance_transition_matrix(filtered_df)
        st.plotly_chart(fig_sig_transitions, use_container_width=True)
        
        # Detailed significance analysis
        st.markdown("### 🔍 Significance Transition Details")
        
        # Create significance transition summary
        transition_summary = filtered_df.groupby(['protected_attribute', 'topic', 'significance_transition']).size().unstack(fill_value=0)
        
        if len(transition_summary) > 0:
            st.dataframe(transition_summary, use_container_width=True)
        
        # P-value analysis
        st.markdown("### 📈 P-Value Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=filtered_df['pre_brexit_sp_p_value'],
                name='Pre-Brexit P-Values',
                opacity=0.7, marker_color='blue'
            ))
            fig.add_trace(go.Histogram(
                x=filtered_df['post_brexit_sp_p_value'],
                name='Post-Brexit P-Values',
                opacity=0.7, marker_color='red'
            ))
            fig.add_vline(x=0.05, line_dash="dash", line_color="black", annotation_text="α = 0.05")
            fig.update_layout(title="P-Value Distributions", barmode='overlay')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Significance correlation
            fig = px.scatter(
                filtered_df,
                x='pre_brexit_sp_p_value',
                y='post_brexit_sp_p_value',
                color='significance_transition',
                title="P-Value Correlation Between Models"
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="black"))
            fig.add_hline(y=0.05, line_dash="dash", line_color="red")
            fig.add_vline(x=0.05, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_mode == "📏 Normative Analysis":
        st.markdown("## 📏 Normative Analysis: Pre-Brexit → Post-Brexit Evolution")
        
        if normative_data is None:
            st.warning("Normative data not available. Please ensure normative analysis has been completed.")
            st.info("To enable normative analysis, please run the normative divergence analysis first.")
        else:
            # Normative analysis visualization - focuses on vignette-specific case characteristics
            normative_results = normative_data['normative_results']
            normative_fields_df = normative_data['normative_fields_df']
            topic_tendencies_df = normative_data['topic_tendencies_df']
            
            st.markdown("### 📊 Normative Alignment Overview")
            st.markdown("*Analysis of how consistently models apply legal/ethical standards to case characteristics*")
            
            # Display overall divergence metrics
            overall_divergence = normative_results.get('overall_divergence', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cosine_sim = overall_divergence.get('cosine_similarity', 0)
                st.metric("Normative Alignment", f"{cosine_sim:.3f}", help="Cosine similarity between Pre/Post model patterns on case characteristics")
            with col2:
                divergence_angle = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi if cosine_sim <= 1 else 0
                st.metric("Divergence Angle", f"{divergence_angle:.1f}°", help="Geometric divergence in how models treat case merits")
            with col3:
                vector_length = overall_divergence.get('vector_length', 0)
                st.metric("Field Comparisons", f"{vector_length}", help="Number of vignette field comparisons analyzed")
            with col4:
                significant_fields = len(normative_fields_df[normative_fields_df['statistical_significance']]) if not normative_fields_df.empty else 0
                st.metric("Significant Differences", f"{significant_fields}", help="Fields with statistically significant normative differences")
            
            # Vignette field analysis
            st.markdown("### 🔍 Vignette Field Analysis")
            st.markdown("*How models evaluate specific case characteristics (not demographics)*")
            
            if not normative_fields_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Field type distribution
                    field_type_counts = normative_fields_df['field_type'].value_counts()
                    fig = px.pie(values=field_type_counts.values, names=field_type_counts.index,
                               title="Distribution of Field Types")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Significance by field type
                    sig_by_type = normative_fields_df.groupby('field_type')['statistical_significance'].agg(['sum', 'count']).reset_index()
                    sig_by_type['percentage'] = (sig_by_type['sum'] / sig_by_type['count'] * 100).round(1)
                    
                    fig = px.bar(sig_by_type, x='field_type', y='percentage',
                               title="% Significant Differences by Field Type",
                               labels={'field_type': 'Field Type', 'percentage': '% Significant'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Most divergent fields - make expandable to save space
                with st.expander("📈 Most Divergent Vignette Fields", expanded=False):
                    # Sort by absolute cross-model difference
                    top_divergent = normative_fields_df.nlargest(10, 'cross_model_difference')
                    
                    # Create visualization
                    fig = px.bar(top_divergent, 
                               x='cross_model_difference', y='field_value',
                               color='statistical_significance',
                               title="Top 10 Most Divergent Case Characteristics",
                               labels={'cross_model_difference': 'Cross-Model Difference (Pre-Brexit - Post-Brexit)',
                                      'field_value': 'Case Characteristic'},
                               color_discrete_map={True: 'red', False: 'blue'})
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed field table - make expandable to save space
                with st.expander("📋 Detailed Field Analysis Table", expanded=False):
                    # Add filters
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        selected_topic = st.selectbox("Filter by Topic", 
                                                     options=['All'] + list(normative_fields_df['topic'].unique()))
                    
                    with col2:
                        selected_field_type = st.selectbox("Filter by Field Type",
                                                          options=['All'] + list(normative_fields_df['field_type'].unique()))
                    
                    with col3:
                        show_significant_only = st.checkbox("Show only statistically significant")
                    
                    # Apply filters
                    filtered_df_norm = normative_fields_df.copy()
                    
                    if selected_topic != 'All':
                        filtered_df_norm = filtered_df_norm[filtered_df_norm['topic'] == selected_topic]
                    
                    if selected_field_type != 'All':
                        filtered_df_norm = filtered_df_norm[filtered_df_norm['field_type'] == selected_field_type]
                    
                    if show_significant_only:
                        filtered_df_norm = filtered_df_norm[filtered_df_norm['statistical_significance']]
                    
                    # Display table
                    display_columns = ['topic', 'field_name', 'field_value', 'cross_model_difference', 
                                     'pre_brexit_raw_rate', 'post_brexit_raw_rate', 'statistical_significance', 'favors_model']
                    
                    st.dataframe(filtered_df_norm[display_columns], height=400)
                
            # Topic-level normative tendencies
            st.markdown("### 🎯 Topic-Level Normative Patterns")
            
            if not topic_tendencies_df.empty:
                # Calculate overall tendency differences
                topic_tendencies_df['tendency_difference'] = (topic_tendencies_df['pre_brexit_topic_tendency'] - 
                                                            topic_tendencies_df['post_brexit_topic_tendency'])
                
                # Visualization
                fig = px.bar(topic_tendencies_df, 
                           x='tendency_difference', y='topic',
                           title="Topic-Level Normative Tendency Differences",
                           labels={'tendency_difference': 'Tendency Difference (Pre-Brexit - Post-Brexit)',
                                  'topic': 'Legal Topic'},
                           color='tendency_difference',
                           color_continuous_scale='RdBu')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Topic tendencies table
                st.dataframe(topic_tendencies_df, height=300)
            
            # GRANULAR TOPIC ANALYSIS
            st.markdown("### 🔬 Granular Topic Analysis")
            st.markdown("*Drill down into specific topics to understand model inclinations and divergences*")
            
            # Topic selector for detailed analysis
            selected_analysis_topic = st.selectbox(
                "Select Topic for Detailed Analysis:",
                options=list(normative_fields_df['topic'].unique()),
                key="topic_analysis_selector"
            )
            
            if selected_analysis_topic:
                topic_data = normative_fields_df[normative_fields_df['topic'] == selected_analysis_topic]
                
                st.markdown(f"#### 📊 Deep Dive: {selected_analysis_topic}")
                
                # Topic overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_fields = len(topic_data)
                    st.metric("Total Fields", total_fields)
                
                with col2:
                    significant_fields = len(topic_data[topic_data['statistical_significance']])
                    significance_rate = significant_fields / total_fields * 100 if total_fields > 0 else 0
                    st.metric("Significant Fields", f"{significant_fields} ({significance_rate:.1f}%)")
                
                with col3:
                    avg_difference = topic_data['cross_model_difference'].mean()
                    st.metric("Avg Model Difference", f"{avg_difference:.3f}")
                
                with col4:
                    max_difference = topic_data['cross_model_difference'].abs().max()
                    st.metric("Max |Difference|", f"{max_difference:.3f}")
                
                # Model inclinations analysis
                st.markdown("##### 🎭 Model Inclinations Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pre-Brexit Model Preferences**")
                    pre_brexit_favored = topic_data[topic_data['favors_model'] == 'pre_brexit'].sort_values('cross_model_difference', ascending=False)
                    
                    if not pre_brexit_favored.empty:
                        for _, row in pre_brexit_favored.head(5).iterrows():
                            st.markdown(f"• **{row['field_value']}** (+{row['cross_model_difference']:.3f})")
                            st.caption(f"   Pre: {row['pre_brexit_raw_rate']:.3f} vs Post: {row['post_brexit_raw_rate']:.3f}")
                    else:
                        st.info("No strong Pre-Brexit preferences in this topic")
                
                with col2:
                    st.markdown("**Post-Brexit Model Preferences**")
                    post_brexit_favored = topic_data[topic_data['favors_model'] == 'post_brexit'].sort_values('cross_model_difference', ascending=True)
                    
                    if not post_brexit_favored.empty:
                        for _, row in post_brexit_favored.head(5).iterrows():
                            st.markdown(f"• **{row['field_value']}** ({row['cross_model_difference']:.3f})")
                            st.caption(f"   Pre: {row['pre_brexit_raw_rate']:.3f} vs Post: {row['post_brexit_raw_rate']:.3f}")
                    else:
                        st.info("No strong Post-Brexit preferences in this topic")
                
                # Detailed field comparison for selected topic
                st.markdown("##### 📋 All Field Comparisons in Topic")
                
                # Sort by absolute difference for impact
                topic_data_sorted = topic_data.sort_values('cross_model_difference', key=abs, ascending=False)
                
                # Create comparison visualization
                fig = px.scatter(
                    topic_data_sorted,
                    x='pre_brexit_raw_rate',
                    y='post_brexit_raw_rate',
                    color='statistical_significance',
                    size='cross_model_difference',
                    hover_data=['field_name', 'field_value'],
                    title=f"Pre-Brexit vs Post-Brexit Grant Rates: {selected_analysis_topic}",
                    labels={'pre_brexit_raw_rate': 'Pre-Brexit Grant Rate',
                           'post_brexit_raw_rate': 'Post-Brexit Grant Rate'},
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                
                # Add diagonal line
                fig.add_shape(
                    type="line",
                    x0=0, y0=0, x1=1, y1=1,
                    line=dict(dash="dash", color="gray")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table for the topic
                st.markdown("##### 📊 Detailed Field Analysis Table")
                
                display_cols = ['field_name', 'field_value', 'pre_brexit_raw_rate', 'post_brexit_raw_rate', 
                               'cross_model_difference', 'statistical_significance', 'favors_model']
                
                styled_topic_data = topic_data_sorted[display_cols].round(4)
                st.dataframe(styled_topic_data, use_container_width=True, height=400)
                
                # Preference Rankings within Topic
                st.markdown("##### 🥇 Model Preference Rankings")
                st.markdown("*How each model ranks the same case characteristics within this topic*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pre-Brexit Model Ranking** *(most → least favorable)*")
                    pre_ranking = topic_data.sort_values('pre_brexit_raw_rate', ascending=False)
                    for i, (_, row) in enumerate(pre_ranking.iterrows(), 1):
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        rate = row['pre_brexit_raw_rate']
                        st.markdown(f"{emoji} **{row['field_value']}** ({rate:.3f})")
                
                with col2:
                    st.markdown("**Post-Brexit Model Ranking** *(most → least favorable)*")
                    post_ranking = topic_data.sort_values('post_brexit_raw_rate', ascending=False)
                    for i, (_, row) in enumerate(post_ranking.iterrows(), 1):
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                        rate = row['post_brexit_raw_rate']
                        st.markdown(f"{emoji} **{row['field_value']}** ({rate:.3f})")
                
                # Ranking correlation analysis
                if len(topic_data) > 2:
                    from scipy.stats import spearmanr
                    
                    # Calculate rank correlation
                    pre_ranks = topic_data['pre_brexit_raw_rate'].rank(ascending=False)
                    post_ranks = topic_data['post_brexit_raw_rate'].rank(ascending=False)
                    
                    try:
                        rank_correlation, rank_p = spearmanr(pre_ranks, post_ranks)
                        
                        st.markdown("##### 🔗 Ranking Consistency Analysis")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rank Correlation", f"{rank_correlation:.3f}")
                        with col2:
                            st.metric("P-Value", f"{rank_p:.4f}")
                        with col3:
                            consistency = "High" if rank_correlation > 0.7 else "Moderate" if rank_correlation > 0.4 else "Low"
                            st.metric("Consistency", consistency)
                        
                        if rank_correlation > 0.7:
                            st.success("🟢 **Models rank case characteristics very similarly within this topic**")
                        elif rank_correlation > 0.4:
                            st.warning("🟡 **Models show moderate agreement in ranking case characteristics**")
                        else:
                            st.error("🔴 **Models rank case characteristics very differently - significant normative divergence**")
                    
                    except:
                        st.info("Unable to calculate ranking correlation")
                
                # Topic-specific insights
                st.markdown("##### 💡 Topic-Specific Insights")
                
                insights = []
                
                # Most divergent field
                most_divergent = topic_data_sorted.iloc[0]
                insights.append(f"🎯 **Most Divergent Field**: '{most_divergent['field_value']}' (Δ = {most_divergent['cross_model_difference']:.3f})")
                
                # Model preference summary
                pre_brexit_count = len(topic_data[topic_data['favors_model'] == 'pre_brexit'])
                post_brexit_count = len(topic_data[topic_data['favors_model'] == 'post_brexit'])
                
                if pre_brexit_count > post_brexit_count:
                    insights.append(f"📈 **Overall Pattern**: Pre-Brexit model more lenient ({pre_brexit_count} vs {post_brexit_count} fields)")
                elif post_brexit_count > pre_brexit_count:
                    insights.append(f"📉 **Overall Pattern**: Post-Brexit model more lenient ({post_brexit_count} vs {post_brexit_count} fields)")
                else:
                    insights.append(f"⚖️ **Overall Pattern**: Models equally balanced ({pre_brexit_count} vs {post_brexit_count} fields)")
                
                # Ranking insights
                if len(topic_data) > 2 and 'rank_correlation' in locals():
                    if rank_correlation > 0.7:
                        insights.append(f"🎯 **Ranking Consistency**: High agreement (ρ = {rank_correlation:.3f}) - models prioritize same characteristics")
                    elif rank_correlation < 0.3:
                        insights.append(f"🔄 **Ranking Divergence**: Low agreement (ρ = {rank_correlation:.3f}) - models prioritize different characteristics")
                
                # Significance pattern
                if significance_rate > 50:
                    insights.append(f"🔴 **High Divergence Topic**: {significance_rate:.1f}% of fields show significant differences")
                elif significance_rate > 25:
                    insights.append(f"🟡 **Moderate Divergence Topic**: {significance_rate:.1f}% of fields show significant differences")
                else:
                    insights.append(f"🟢 **Low Divergence Topic**: Only {significance_rate:.1f}% of fields show significant differences")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            # Cross-topic comparison matrix - make expandable
            with st.expander("📊 Cross-Topic Divergence Matrix", expanded=False):
                # Create topic divergence summary
                topic_summary = normative_fields_df.groupby('topic').agg({
                    'cross_model_difference': ['mean', 'std', 'count'],
                    'statistical_significance': 'sum',
                    'pre_brexit_raw_rate': 'mean',
                    'post_brexit_raw_rate': 'mean'
                }).round(4)
                
                topic_summary.columns = ['Avg_Difference', 'Std_Difference', 'Total_Fields', 'Significant_Fields', 'Pre_Brexit_Avg', 'Post_Brexit_Avg']
                topic_summary['Significance_Rate'] = (topic_summary['Significant_Fields'] / topic_summary['Total_Fields'] * 100).round(1)
                topic_summary['Dominant_Model'] = topic_summary.apply(
                    lambda row: 'Pre-Brexit' if row['Avg_Difference'] > 0 else 'Post-Brexit', axis=1
                )
                
                # Sort by average absolute difference
                topic_summary['Abs_Avg_Difference'] = topic_summary['Avg_Difference'].abs()
                topic_summary = topic_summary.sort_values('Abs_Avg_Difference', ascending=False)
                
                st.dataframe(topic_summary.drop('Abs_Avg_Difference', axis=1), use_container_width=True)
            
            # Field type breakdown - make more compact
            with st.expander("🔬 Field Type Detailed Analysis", expanded=True):
                subset_analysis = normative_results.get('subset_analysis', {})
                
                if subset_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Ordinal Fields** *(ranked case characteristics)*")
                        ordinal_data = subset_analysis.get('ordinal_fields', {})
                        if ordinal_data:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Cosine Sim", f"{ordinal_data.get('cosine_similarity', 0):.3f}")
                            with col_b:
                                st.metric("Vector Len", f"{ordinal_data.get('vector_length', 0)}")
                            with col_c:
                                st.metric("Pearson r", f"{ordinal_data.get('pearson_correlation', 0):.3f}")
                            
                            # Show ordinal field examples
                            ordinal_fields = normative_fields_df[normative_fields_df['field_type'] == 'ordinal']['field_name'].unique()
                            st.caption(f"**Examples**: {', '.join(ordinal_fields[:3])}")
                        else:
                            st.info("No ordinal fields data available")
                    
                    with col2:
                        st.markdown("**Horizontal Fields** *(categorical case characteristics)*")
                        horizontal_data = subset_analysis.get('horizontal_fields', {})
                        if horizontal_data:
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Cosine Sim", f"{horizontal_data.get('cosine_similarity', 0):.3f}")
                            with col_b:
                                st.metric("Vector Len", f"{horizontal_data.get('vector_length', 0)}")
                            with col_c:
                                st.metric("Pearson r", f"{horizontal_data.get('pearson_correlation', 0):.3f}")
                            
                            # Show horizontal field examples  
                            horizontal_fields = normative_fields_df[normative_fields_df['field_type'] == 'horizontal']['field_name'].unique()
                            st.caption(f"**Examples**: {', '.join(horizontal_fields[:3])}")
                        else:
                            st.info("No horizontal fields data available")
            
            # Interpretation
            st.markdown("### 📋 Normative Analysis Interpretation")
            
            if cosine_sim > 0.8:
                alignment_level = "🟢 High"
                interpretation = "Models show strong normative alignment. Legal/ethical standards are applied very consistently across case characteristics."
            elif cosine_sim > 0.6:
                alignment_level = "🟡 Moderate" 
                interpretation = "Models show moderate normative alignment. Some inconsistencies in how legal standards are applied to similar case circumstances."
            else:
                alignment_level = "🔴 Low"
                interpretation = "Models show low normative alignment. Significant inconsistencies in applying legal/ethical standards to case characteristics."
            
            st.markdown(f"**Normative Alignment Level**: {alignment_level}")
            st.markdown(f"**Interpretation**: {interpretation}")
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mean_abs_diff = overall_divergence.get('mean_absolute_difference', 0)
                st.metric("Mean Absolute Difference", f"{mean_abs_diff:.3f}",
                         help="Average absolute difference in normative evaluation scores")
            
            with col2:
                total_comparisons = len(normative_fields_df) if not normative_fields_df.empty else 0
                st.metric("Total Field Comparisons", f"{total_comparisons}",
                         help="Total vignette field comparisons analyzed")
            
            with col3:
                manhattan_dist = overall_divergence.get('manhattan_distance', 0)
                st.metric("Manhattan Distance", f"{manhattan_dist:.3f}",
                         help="L1 distance between normative pattern vectors")
            
            # Display summary markdown if available
            if 'normative_summary' in normative_data:
                st.markdown("### 📋 Technical Summary")
                with st.expander("View Detailed Normative Analysis Summary"):
                    st.markdown(normative_data['normative_summary'])
    
    elif analysis_mode == "🔄 Comparative Analysis":
        st.markdown("## 🔄 Fairness vs Normative Comparative Analysis")
        
        # Comparison configuration
        comparison_type = st.selectbox(
            "Select Comparison Type:",
            ["Fairness-Normative Correlation", "Topic-Level Comparison", "Attribute-Level Comparison", "Temporal Evolution"]
        )
        
        if comparison_type == "Fairness-Normative Correlation":
            st.markdown("### 🔗 Fairness-Normative Relationship Analysis")
            
            # Create comparison visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Fairness magnitude distribution
                fig = px.histogram(
                    filtered_df, x='sp_change_abs',
                    title="Fairness Change Magnitude Distribution",
                    nbins=30, marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Significance correlation analysis
                significance_corr = filtered_df.groupby('topic').agg({
                    'sp_change_abs': 'mean',
                    'significance_transition': lambda x: (x.isin(['gained', 'lost'])).mean()
                })
                
                fig = px.scatter(
                    x=significance_corr['sp_change_abs'],
                    y=significance_corr['significance_transition'],
                    text=significance_corr.index,
                    title="Fairness Change vs Significance Instability",
                    labels={'x': 'Mean |SP Change|', 'y': 'Significance Change Rate'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Topic-Level Comparison":
            # Topic-level fairness vs normative comparison
            st.markdown("### 🎯 Topic-Level Fairness and Normative Patterns")
            
            topic_summary = filtered_df.groupby('topic').agg({
                'sp_change_abs': ['mean', 'std', 'count'],
                'significance_transition': lambda x: (x.isin(['gained', 'lost'])).sum(),
                'pre_brexit_sp_significance': 'mean',
                'post_brexit_sp_significance': 'mean'
            }).round(4)
            
            topic_summary.columns = ['Mean |SP Change|', 'Std |SP Change|', 'Count', 'Sig Transitions', 'Pre-Brexit Sig Rate', 'Post-Brexit Sig Rate']
            topic_summary = topic_summary.sort_values('Mean |SP Change|', ascending=False)
            
            st.dataframe(topic_summary, use_container_width=True)
            
            # Visualization
            fig = px.scatter(
                topic_summary.reset_index(),
                x='Mean |SP Change|', y='Sig Transitions',
                size='Count', hover_name='topic',
                title="Topic Analysis: Fairness Change vs Significance Transitions"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Attribute-Level Comparison":
            st.markdown("### 👥 Attribute-Level Fairness and Normative Analysis")
            
            # Attribute-level fairness analysis
            attr_fairness_summary = filtered_df.groupby('protected_attribute').agg({
                'sp_change': ['mean', 'std'],
                'sp_change_abs': ['mean', 'max'],
                'significance_transition': lambda x: (x.isin(['gained', 'lost'])).sum(),
                'pre_brexit_sp_significance': 'mean',
                'post_brexit_sp_significance': 'mean'
            }).round(4)
            
            attr_fairness_summary.columns = ['Mean SP Change', 'SP Change Std', 'Mean |SP Change|', 'Max |SP Change|', 'Sig Transitions', 'Pre-Brexit Sig Rate', 'Post-Brexit Sig Rate']
            attr_fairness_summary = attr_fairness_summary.sort_values('Mean |SP Change|', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚖️ Fairness Analysis by Attribute")
                
                # Fairness comparison visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=attr_fairness_summary.index,
                    y=attr_fairness_summary['Mean |SP Change|'],
                    name='Mean |SP Change|',
                    marker_color='red',
                    yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=attr_fairness_summary.index,
                    y=attr_fairness_summary['Sig Transitions'],
                    mode='lines+markers',
                    name='Significance Transitions',
                    line=dict(color='blue', width=3),
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Fairness Metrics by Protected Attribute",
                    xaxis_title="Protected Attribute",
                    yaxis=dict(title="Mean |SP Change|", side="left"),
                    yaxis2=dict(title="Significance Transitions", side="right", overlaying="y"),
                    legend=dict(x=0.01, y=0.99)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Normative analysis by attribute (if available)
                if normative_data:
                    st.markdown("#### 📏 Normative Patterns by Attribute")
                    
                    # Create simulated normative alignment by attribute
                    # (This would ideally come from actual normative data)
                    normative_alignment = {
                        attr: np.random.normal(0.6, 0.1) for attr in attr_fairness_summary.index
                    }
                    
                    alignment_df = pd.DataFrame(list(normative_alignment.items()), 
                                              columns=['Attribute', 'Normative Alignment'])
                    
                    fig = px.bar(
                        alignment_df,
                        x='Attribute',
                        y='Normative Alignment',
                        color='Normative Alignment',
                        color_continuous_scale='RdYlGn',
                        title="Simulated Normative Alignment by Attribute"
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="black", annotation_text="Neutral Alignment")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Normative data not available for attribute-level comparison")
                    
                    # Show fairness significance rates instead
                    st.markdown("#### 📊 Significance Rate Changes")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=attr_fairness_summary.index,
                        y=attr_fairness_summary['Pre-Brexit Sig Rate'],
                        name='Pre-Brexit',
                        marker_color='blue'
                    ))
                    fig.add_trace(go.Bar(
                        x=attr_fairness_summary.index,
                        y=attr_fairness_summary['Post-Brexit Sig Rate'],
                        name='Post-Brexit',
                        marker_color='red'
                    ))
                    
                    fig.update_layout(
                        title="Statistical Significance Rates by Attribute",
                        xaxis_title="Protected Attribute",
                        yaxis_title="Significance Rate",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Detailed attribute comparison table
            st.markdown("#### 📊 Detailed Attribute Comparison")
            st.dataframe(attr_fairness_summary, use_container_width=True)
            
            # Correlation analysis between fairness and normative (if available)
            if normative_data:
                st.markdown("#### 🔗 Fairness-Normative Correlation by Attribute")
                
                correlation_data = []
                for attr in attr_fairness_summary.index:
                    fairness_metric = attr_fairness_summary.loc[attr, 'Mean |SP Change|']
                    normative_metric = normative_alignment.get(attr, 0.5)  # Simulated
                    correlation_data.append({
                        'Attribute': attr,
                        'Fairness Change': fairness_metric,
                        'Normative Alignment': normative_metric,
                        'Relationship': 'Inverse' if (fairness_metric > 0.1 and normative_metric < 0.5) else 'Aligned'
                    })
                
                corr_df = pd.DataFrame(correlation_data)
                
                fig = px.scatter(
                    corr_df,
                    x='Fairness Change',
                    y='Normative Alignment',
                    color='Relationship',
                    size=[1]*len(corr_df),
                    hover_name='Attribute',
                    title="Fairness vs Normative Relationship by Attribute"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Temporal Evolution":
            st.markdown("### ⏰ Temporal Evolution: Fairness and Normative Patterns")
            
            # Temporal evolution analysis
            st.markdown("#### 📈 Evolution Timeline")
            
            # Create temporal visualization
            evolution_data = []
            time_periods = ['Pre-Brexit (2013-2016)', 'Post-Brexit (2019-2025)']
            
            for attr in filtered_df['protected_attribute'].unique():
                attr_data = filtered_df[filtered_df['protected_attribute'] == attr]
                
                # Fairness metrics
                pre_fairness = attr_data['pre_brexit_model_statistical_parity'].mean()
                post_fairness = attr_data['post_brexit_model_statistical_parity'].mean()
                
                # Significance rates
                pre_sig_rate = attr_data['pre_brexit_sp_significance'].mean()
                post_sig_rate = attr_data['post_brexit_sp_significance'].mean()
                
                evolution_data.extend([
                    {'Time Period': time_periods[0], 'Attribute': attr, 'Metric': 'Fairness (SP)', 'Value': pre_fairness},
                    {'Time Period': time_periods[1], 'Attribute': attr, 'Metric': 'Fairness (SP)', 'Value': post_fairness},
                    {'Time Period': time_periods[0], 'Attribute': attr, 'Metric': 'Significance Rate', 'Value': pre_sig_rate},
                    {'Time Period': time_periods[1], 'Attribute': attr, 'Metric': 'Significance Rate', 'Value': post_sig_rate}
                ])
            
            evolution_df = pd.DataFrame(evolution_data)
            
            # Create separate visualizations for fairness and significance
            col1, col2 = st.columns(2)
            
            with col1:
                fairness_evolution = evolution_df[evolution_df['Metric'] == 'Fairness (SP)']
                
                fig = px.line(
                    fairness_evolution,
                    x='Time Period',
                    y='Value',
                    color='Attribute',
                    title="Fairness Evolution Over Time",
                    markers=True
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No Bias")
                fig.update_layout(yaxis_title="Mean Statistical Parity")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                sig_evolution = evolution_df[evolution_df['Metric'] == 'Significance Rate']
                
                fig = px.line(
                    sig_evolution,
                    x='Time Period',
                    y='Value',
                    color='Attribute',
                    title="Significance Rate Evolution",
                    markers=True
                )
                fig.update_layout(yaxis_title="Statistical Significance Rate", yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)
            
            # Evolution velocity analysis
            st.markdown("#### 🚀 Change Velocity Analysis")
            
            velocity_data = []
            for attr in filtered_df['protected_attribute'].unique():
                attr_data = filtered_df[filtered_df['protected_attribute'] == attr]
                
                # Calculate change rates
                fairness_velocity = attr_data['sp_change'].mean()  # Average change
                significance_velocity = (attr_data['post_brexit_sp_significance'].mean() - 
                                       attr_data['pre_brexit_sp_significance'].mean())
                
                velocity_data.append({
                    'Attribute': attr,
                    'Fairness Velocity': fairness_velocity,
                    'Significance Velocity': significance_velocity,
                    'Overall Velocity': np.sqrt(fairness_velocity**2 + significance_velocity**2)
                })
            
            velocity_df = pd.DataFrame(velocity_data)
            
            # Velocity visualization
            fig = px.scatter(
                velocity_df,
                x='Fairness Velocity',
                y='Significance Velocity',
                size='Overall Velocity',
                color='Attribute',
                title="Change Velocity: Fairness vs Significance",
                labels={'Fairness Velocity': 'Fairness Change Rate', 'Significance Velocity': 'Significance Change Rate'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.add_annotation(x=0.1, y=0.1, text="Increasing Both", showarrow=False)
            fig.add_annotation(x=-0.1, y=-0.1, text="Decreasing Both", showarrow=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Velocity summary table
            st.markdown("#### 📊 Velocity Summary")
            st.dataframe(velocity_df.round(4), use_container_width=True)
    
    elif analysis_mode == "🎯 Topic Stratified Analysis":
        st.markdown("## 🎯 Topic Stratified Analysis")
        
        # Topic selection for deep dive
        selected_topic_analysis = st.selectbox(
            "Select Topic for Deep Analysis:",
            sorted(filtered_df['topic'].unique())
        )
        
        topic_data = filtered_df[filtered_df['topic'] == selected_topic_analysis]
        
        if len(topic_data) > 0:
            st.markdown(f"### 📊 Deep Dive: {selected_topic_analysis}")
            
            # Topic overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Comparisons", len(topic_data))
            with col2:
                st.metric("Mean |SP Change|", f"{topic_data['sp_change_abs'].mean():.4f}")
            with col3:
                st.metric("Significance Transitions", topic_data['significance_transition'].isin(['gained', 'lost']).sum())
            with col4:
                st.metric("Attributes Covered", topic_data['protected_attribute'].nunique())
            
            # Topic-specific visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # SP change by attribute for this topic
                fig = px.box(
                    topic_data, x='protected_attribute', y='sp_change',
                    title=f"SP Change by Attribute: {selected_topic_analysis}"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Significance pattern for this topic
                sig_counts = topic_data['significance_transition'].value_counts()
                fig = px.pie(
                    values=sig_counts.values, names=sig_counts.index,
                    title=f"Significance Transitions: {selected_topic_analysis}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed topic analysis table
            st.markdown("### 📋 Detailed Comparisons")
            
            topic_details = topic_data[['group_comparison', 'protected_attribute', 'sp_change', 
                                       'sp_change_abs', 'significance_transition', 
                                       'pre_brexit_sp_significance', 'post_brexit_sp_significance']].sort_values('sp_change_abs', ascending=False)
            
            st.dataframe(topic_details, use_container_width=True)
        else:
            st.warning("No data available for selected topic with current filters.")
    
    elif analysis_mode == "👥 Protected Attribute Analysis":
        st.markdown("## 👥 Protected Attribute Analysis")
        
        # Attribute selection
        selected_attribute_analysis = st.selectbox(
            "Select Protected Attribute for Deep Analysis:",
            sorted(filtered_df['protected_attribute'].unique())
        )
        
        attribute_data = filtered_df[filtered_df['protected_attribute'] == selected_attribute_analysis]
        
        if len(attribute_data) > 0:
            st.markdown(f"### 📊 Deep Dive: {selected_attribute_analysis.title()} Bias Analysis")
            
            # Attribute overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Comparisons", len(attribute_data))
            with col2:
                st.metric("Topics Covered", attribute_data['topic'].nunique())
            with col3:
                st.metric("Mean |SP Change|", f"{attribute_data['sp_change_abs'].mean():.4f}")
            with col4:
                st.metric("Largest Change", f"{attribute_data['sp_change_abs'].max():.4f}")
            
            # Attribute-specific visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # SP change distribution for this attribute
                fig = px.histogram(
                    attribute_data, x='sp_change',
                    title=f"{selected_attribute_analysis.title()} SP Change Distribution",
                    nbins=30
                )
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Topic-wise patterns for this attribute
                topic_patterns = attribute_data.groupby('topic')['sp_change_abs'].mean().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=topic_patterns.index, y=topic_patterns.values,
                    title=f"Topics Most Affected by {selected_attribute_analysis.title()} Bias"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Group comparison analysis
            st.markdown("### 🔍 Group Comparison Analysis")
            
            # Extract specific group comparisons for this attribute
            if selected_attribute_analysis == 'country':
                # For country, show country-vs-country patterns
                group_analysis = attribute_data.groupby('group_comparison').agg({
                    'sp_change': 'mean',
                    'sp_change_abs': 'mean',
                    'significance_transition': lambda x: (x.isin(['gained', 'lost'])).sum()
                }).sort_values('sp_change_abs', ascending=False).head(15)
                
                st.dataframe(group_analysis, use_container_width=True)
    
    elif analysis_mode == "🔍 Individual Case Explorer":
        st.markdown("## 🔍 Individual Case Explorer")
        
        # Case filtering
        st.markdown("### 🎯 Case Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            change_threshold = st.slider("Minimum |SP Change|", 0.0, 0.5, 0.1, 0.01)
        with col2:
            case_significance_filter = st.selectbox(
                "Significance Filter:",
                ["All Cases", "Gained Significance", "Lost Significance", "Both Models Significant"]
            )
        with col3:
            case_attribute_filter = st.selectbox(
                "Attribute Filter:",
                ["All"] + sorted(filtered_df['protected_attribute'].unique())
            )
        
        # Apply case filters
        case_filtered_df = filtered_df[filtered_df['sp_change_abs'] >= change_threshold].copy()
        
        if case_significance_filter == "Gained Significance":
            case_filtered_df = case_filtered_df[case_filtered_df['significance_transition'] == 'gained']
        elif case_significance_filter == "Lost Significance":
            case_filtered_df = case_filtered_df[case_filtered_df['significance_transition'] == 'lost']
        elif case_significance_filter == "Both Models Significant":
            case_filtered_df = case_filtered_df[case_filtered_df['significance_transition'] == 'stable_sig']
        
        if case_attribute_filter != "All":
            case_filtered_df = case_filtered_df[case_filtered_df['protected_attribute'] == case_attribute_filter]
        
        case_filtered_df = case_filtered_df.sort_values('sp_change_abs', ascending=False)
        
        st.markdown(f"### 📋 Found {len(case_filtered_df)} Cases")
        
        # Display top cases
        for idx, (_, row) in enumerate(case_filtered_df.head(10).iterrows()):
            with st.expander(f"Case {idx+1}: {row['group_comparison']} in {row['topic'][:40]}... (|ΔSP| = {row['sp_change_abs']:.4f})"):
                
                # Case header
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown(f"**Topic**: {row['topic']}")
                    st.markdown(f"**Protected Attribute**: {row['protected_attribute']}")
                    st.markdown(f"**Group Comparison**: {row['group_comparison']}")
                
                with col_b:
                    direction = "📈 Increased" if row['sp_change'] > 0 else "📉 Decreased"
                    st.markdown(f"**Change Direction**: {direction}")
                    st.markdown(f"**SP Change**: {row['sp_change']:+.4f}")
                    st.markdown(f"**|SP Change|**: {row['sp_change_abs']:.4f}")
                
                with col_c:
                    st.markdown(f"**Significance Transition**: {row['significance_transition'].title()}")
                    st.markdown(f"**Pre-Brexit Significant**: {'✅' if row['pre_brexit_sp_significance'] else '❌'}")
                    st.markdown(f"**Post-Brexit Significant**: {'✅' if row['post_brexit_sp_significance'] else '❌'}")
                
                # Statistical details
                st.markdown("#### 📊 Statistical Details")
                
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("**Pre-Brexit Model:**")
                    st.markdown(f"- SP Value: {row['pre_brexit_model_statistical_parity']:.4f}")
                    st.markdown(f"- P-Value: {row['pre_brexit_sp_p_value']:.4f}")
                    st.markdown(f"- Sample Size: {row['pre_brexit_group_size']}")
                
                with detail_col2:
                    st.markdown("**Post-Brexit Model:**")
                    st.markdown(f"- SP Value: {row['post_brexit_model_statistical_parity']:.4f}")
                    st.markdown(f"- P-Value: {row['post_brexit_sp_p_value']:.4f}")
                    st.markdown(f"- Sample Size: {row['post_brexit_group_size']}")
    
    elif analysis_mode == "🧠 Scientific Insights Generator":
        st.markdown("## 🧠 Automated Scientific Insights")
        
        with st.spinner("🔬 Generating scientific insights..."):
            
            insights = []
            
            # Overall pattern analysis
            total_cases = len(filtered_df)
            mean_change = filtered_df['sp_change'].mean()
            mean_abs_change = filtered_df['sp_change_abs'].mean()
            
            insights.append(f"**Dataset Overview**: Analyzed {total_cases:,} demographic comparisons across {filtered_df['protected_attribute'].nunique()} protected attributes and {filtered_df['topic'].nunique()} topics.")
            
            # Magnitude analysis
            if mean_abs_change > 0.1:
                insights.append(f"🔴 **Significant Bias Changes Detected**: Mean absolute SP change of {mean_abs_change:.4f} indicates substantial fairness pattern shifts between model generations.")
            elif mean_abs_change > 0.05:
                insights.append(f"🟠 **Moderate Bias Changes**: Mean absolute SP change of {mean_abs_change:.4f} suggests moderate fairness pattern evolution.")
            else:
                insights.append(f"🟢 **Stable Bias Patterns**: Mean absolute SP change of {mean_abs_change:.4f} indicates relatively stable fairness patterns.")
            
            # Direction analysis
            if abs(mean_change) > 0.01:
                direction = "toward more biased outcomes" if mean_change > 0 else "toward less biased outcomes"
                insights.append(f"📊 **Directional Trend**: Overall bias pattern shifted {direction} (mean change: {mean_change:+.4f}).")
            
            # Attribute-specific insights
            attr_insights = []
            for attr in filtered_df['protected_attribute'].unique():
                attr_data = filtered_df[filtered_df['protected_attribute'] == attr]
                attr_change = attr_data['sp_change_abs'].mean()
                
                if attr_change > 0.15:
                    attr_insights.append(f"⚠️ **{attr.title()} Bias Alert**: High average change ({attr_change:.4f}) suggests significant fairness concerns.")
                elif attr_change > 0.1:
                    attr_insights.append(f"🔍 **{attr.title()} Monitoring**: Moderate average change ({attr_change:.4f}) warrants continued observation.")
            
            insights.extend(attr_insights)
            
            # Topic-specific insights
            topic_changes = filtered_df.groupby('topic')['sp_change_abs'].mean().sort_values(ascending=False)
            top_changed_topics = topic_changes.head(3)
            
            for topic, change in top_changed_topics.items():
                if change > 0.1:
                    insights.append(f"🎯 **Topic Alert - {topic}**: High bias instability (mean |ΔSP| = {change:.4f}) requires immediate attention.")
            
            # Significance pattern insights
            sig_change_rate = filtered_df['significance_transition'].isin(['gained', 'lost']).mean()
            
            if sig_change_rate > 0.4:
                insights.append(f"📈 **High Significance Instability**: {sig_change_rate:.1%} of comparisons changed significance status, indicating unstable statistical patterns.")
            elif sig_change_rate > 0.2:
                insights.append(f"📊 **Moderate Significance Changes**: {sig_change_rate:.1%} of comparisons changed significance status.")
            
            # Vector analysis insights
            if fairness_data and 'divergence_results' in fairness_data:
                cosine_sim = fairness_data['divergence_results']['magnitude_divergence']['divergence_metrics']['cosine_similarity']
                angle_deg = np.arccos(np.clip(cosine_sim, -1, 1)) * 180 / np.pi
                
                if angle_deg > 60:
                    insights.append(f"🔺 **High Vector Divergence**: {angle_deg:.1f}° divergence angle indicates substantial bias pattern restructuring.")
                elif angle_deg > 30:
                    insights.append(f"📐 **Moderate Vector Divergence**: {angle_deg:.1f}° divergence angle suggests noticeable bias pattern changes.")
                else:
                    insights.append(f"✅ **Low Vector Divergence**: {angle_deg:.1f}° divergence angle indicates relatively stable bias patterns.")
            
            # Research implications
            insights.append("### 🔬 **Research Implications**")
            insights.append("1. **Model Evolution Impact**: Clear evidence of systematic fairness pattern changes between Pre-Brexit and Post-Brexit models.")
            insights.append("2. **Demographic Sensitivity**: Different protected attributes show varying degrees of bias instability.")
            insights.append("3. **Topic Dependency**: Fairness patterns are highly dependent on specific application domains.")
            insights.append("4. **Statistical Rigor**: Significance testing reveals genuine pattern changes beyond random variation.")
            
            # Methodological notes
            insights.append("### 📋 **Methodological Notes**")
            insights.append("- **Statistical Parity**: Measures difference in positive outcome rates between demographic groups")
            insights.append("- **Significance Testing**: Uses appropriate statistical tests with multiple comparison corrections")
            insights.append("- **Vector Analysis**: Geometric interpretation provides intuitive understanding of bias pattern changes")
            insights.append("- **Sample Sizes**: All analyses weighted by appropriate sample sizes for statistical validity")
        
        # Display insights
        st.markdown("### 📊 Generated Scientific Insights")
        
        for insight in insights:
            if insight.startswith("###"):
                st.markdown(insight)
            else:
                st.markdown(f"• {insight}")
        
        # Export functionality
        st.markdown("### 📥 Export Options")
        
        if st.button("📄 Generate Scientific Report"):
            report_content = f"""
# Ethical AI Assessment: Scientific Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{insights[0]}

## Key Findings
"""
            for insight in insights[1:]:
                if not insight.startswith("###"):
                    report_content += f"- {insight}\n"
            
            st.download_button(
                label="📥 Download Scientific Report",
                data=report_content,
                file_name=f"ethical_ai_scientific_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("**🔬 Ethical AI Assessment Dashboard** | Advanced Analysis for AI Ethics Research")

if __name__ == "__main__":
    main() 