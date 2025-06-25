#!/usr/bin/env python3
"""
Simplified Interactive AI Model Evaluation Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="AI Model Evaluation Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_json_file(filepath: str) -> Dict:
    """Load a single JSON file"""
    # Try multiple possible roots
    possible_roots = [
        Path.cwd(),
        Path(__file__).parent.parent,
        Path("/data/shil6369/gov_scraper")
    ]
    
    for root in possible_roots:
        full_path = root / filepath
        if full_path.exists():
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'processed_results' in data:
                results_dict = {}
                for result in data['processed_results']:
                    sample_id = result.get('sample_id')
                    if sample_id:
                        results_dict[str(sample_id)] = result
                return results_dict
    
    raise FileNotFoundError(f"Could not find file: {filepath}")

def create_comparison_data(pre_data: Dict, post_data: Dict) -> pd.DataFrame:
    """Create comparison dataframe"""
    common_ids = set(pre_data.keys()).intersection(set(post_data.keys()))
    
    comparison_results = []
    for sample_id in common_ids:
        pre_result = pre_data[sample_id]
        post_result = post_data[sample_id]
        
        metadata = pre_result.get('metadata', {})
        fields = metadata.get('fields', {})
        
        # Normalize decisions to lowercase
        pre_decision = pre_result.get('decision', '').lower()
        post_decision = post_result.get('decision', '').lower()
        
        comparison = {
            'sample_id': sample_id,
            'topic': metadata.get('topic'),
            'meta_topic': metadata.get('meta_topic'),
            'pre_brexit_decision': pre_decision,
            'post_brexit_decision': post_decision,
            'decisions_match': pre_decision == post_decision,
            'decision_pattern': f'{pre_decision} → {post_decision}',
            'age': fields.get('age', 'Unknown'),
            'religion': fields.get('religion', 'Unknown'),
            'gender': fields.get('gender', 'Unknown'),
            'country': fields.get('country', 'Unknown'),
            'vignette_text': metadata.get('vignette_text', ''),
            'pre_brexit_reasoning': pre_result.get('reasoning', ''),
            'post_brexit_reasoning': post_result.get('reasoning', ''),
        }
        comparison_results.append(comparison)
    
    return pd.DataFrame(comparison_results)

# Main title
st.markdown("# 🔍 AI Model Evaluation Dashboard")

# Sidebar for data loading
st.sidebar.markdown("## 📂 Data Loading")

# File inputs
pre_brexit_file = st.sidebar.text_input(
    "Pre-Brexit File:",
    value="inference/results/processed/processed_production_subset_inference_llama3_8b_pre_brexit_2013_2016_instruct_20250623_194615_20250625_133855/successful_extractions.json"
)

post_brexit_file = st.sidebar.text_input(
    "Post-Brexit File:",
    value="inference/results/processed/processed_production_subset_inference_llama3_8b_post_brexit_2019_2025_instruct_20250623_203131_20250625_134035/successful_extractions.json"
)

# Load data button
if st.sidebar.button("🚀 Load Data", type="primary"):
    try:
        with st.spinner("Loading data..."):
            # Load files
            st.info("Loading pre-Brexit data...")
            pre_data = load_json_file(pre_brexit_file)
            st.info(f"✅ Loaded {len(pre_data)} pre-Brexit cases")
            
            st.info("Loading post-Brexit data...")
            post_data = load_json_file(post_brexit_file)
            st.info(f"✅ Loaded {len(post_data)} post-Brexit cases")
            
            # Create comparison
            st.info("Creating comparison dataset...")
            df = create_comparison_data(pre_data, post_data)
            st.info(f"✅ Found {len(df)} matching cases")
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"🎉 Successfully loaded {len(df)} cases!")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Check if data is loaded
if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
    st.info("👆 Please load data using the sidebar to start analysis")
    st.stop()

# Get data from session state
df = st.session_state.df

# Analysis mode selection
analysis_mode = st.sidebar.selectbox(
    "📊 Analysis Mode",
    ["Overview", "Topic Analysis", "Meta Topic Analysis", "Protected Attributes", "Cross-sectional Analysis", "Individual Cases", "Generate Report"]
)

# Global filters
st.sidebar.markdown("## 🎯 Filters")

# Topic filter
topics = ['All'] + sorted([t for t in df['topic'].unique() if t is not None])
selected_topic = st.sidebar.selectbox("Topic", topics)

# Agreement filter
agreement_filter = st.sidebar.selectbox(
    "Agreement Status",
    ["All", "Agreement Only", "Disagreement Only"]
)

# Apply filters
filtered_df = df.copy()

if selected_topic != 'All':
    filtered_df = filtered_df[filtered_df['topic'] == selected_topic]

if agreement_filter == 'Agreement Only':
    filtered_df = filtered_df[filtered_df['decisions_match'] == True]
elif agreement_filter == 'Disagreement Only':
    filtered_df = filtered_df[filtered_df['decisions_match'] == False]

# Overview metrics
st.markdown("## 📊 Overview Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

total_cases = len(filtered_df)
agreement_rate = filtered_df['decisions_match'].mean() if len(filtered_df) > 0 else 0
pre_grant_rate = (filtered_df['pre_brexit_decision'] == 'granted').mean() if len(filtered_df) > 0 else 0
post_grant_rate = (filtered_df['post_brexit_decision'] == 'granted').mean() if len(filtered_df) > 0 else 0
disagreement_cases = len(filtered_df[filtered_df['decisions_match'] == False])

with col1:
    st.metric("Total Cases", f"{total_cases:,}")
with col2:
    st.metric("Agreement Rate", f"{agreement_rate:.1%}")
with col3:
    st.metric("Pre-Brexit Grant Rate", f"{pre_grant_rate:.1%}")
with col4:
    st.metric("Post-Brexit Grant Rate", f"{post_grant_rate:.1%}")
with col5:
    st.metric("Disagreement Cases", f"{disagreement_cases:,}")

# Analysis sections based on mode
if analysis_mode == "Overview":
    st.markdown("## 🎯 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Decision distribution
        decision_counts = filtered_df['decision_pattern'].value_counts()
        fig = px.pie(values=decision_counts.values, names=decision_counts.index,
                     title="Decision Pattern Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Agreement by topic
        topic_agreement = filtered_df.groupby('topic')['decisions_match'].agg(['mean', 'count']).reset_index()
        topic_agreement = topic_agreement[topic_agreement['count'] >= 5]  # Filter small groups
        
        fig = px.bar(topic_agreement, x='topic', y='mean',
                     title="Agreement Rate by Topic",
                     labels={'mean': 'Agreement Rate'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

elif analysis_mode == "Meta Topic Analysis":
    st.markdown("## 🎯 Meta Topic Analysis")
    
    if len(filtered_df) == 0:
        st.warning("No data available with current filters.")
    else:
        # Meta topic overview
        meta_topic_stats = []
        for meta_topic in filtered_df['meta_topic'].unique():
            if meta_topic is None:
                continue
            
            meta_data = filtered_df[filtered_df['meta_topic'] == meta_topic]
            stats = {
                'Meta Topic': meta_topic,
                'Cases': len(meta_data),
                'Agreement Rate': meta_data['decisions_match'].mean(),
                'Pre-Brexit Grant Rate': (meta_data['pre_brexit_decision'] == 'granted').mean(),
                'Post-Brexit Grant Rate': (meta_data['post_brexit_decision'] == 'granted').mean(),
                'Grant Rate Difference': (meta_data['post_brexit_decision'] == 'granted').mean() - (meta_data['pre_brexit_decision'] == 'granted').mean(),
                'Topics Count': meta_data['topic'].nunique()
            }
            meta_topic_stats.append(stats)
        
        meta_df = pd.DataFrame(meta_topic_stats).sort_values('Cases', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Meta topic agreement rates
            fig = px.bar(meta_df, x='Meta Topic', y='Agreement Rate',
                         color='Agreement Rate', color_continuous_scale='RdYlGn',
                         title="Agreement Rates by Meta Topic")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Grant rate comparison by meta topic
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Pre-Brexit', x=meta_df['Meta Topic'], 
                                y=meta_df['Pre-Brexit Grant Rate'], marker_color='blue'))
            fig.add_trace(go.Bar(name='Post-Brexit', x=meta_df['Meta Topic'], 
                                y=meta_df['Post-Brexit Grant Rate'], marker_color='red'))
            fig.update_layout(title="Grant Rates by Meta Topic", barmode='group')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Grant rate changes
        st.markdown("### 📈 Grant Rate Changes (Post-Brexit vs Pre-Brexit)")
        meta_df_sorted = meta_df.sort_values('Grant Rate Difference', ascending=False)
        
        fig = px.bar(meta_df_sorted, x='Meta Topic', y='Grant Rate Difference',
                     color='Grant Rate Difference', color_continuous_scale='RdBu',
                     title="Grant Rate Change by Meta Topic (Post-Brexit - Pre-Brexit)")
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics table
        st.markdown("### 📊 Meta Topic Statistics")
        st.dataframe(meta_df.round(3), use_container_width=True)
        
        # Meta topic insights
        st.markdown("### 💡 Key Insights")
        if len(meta_df) > 0:
            highest_agreement = meta_df.loc[meta_df['Agreement Rate'].idxmax()]
            lowest_agreement = meta_df.loc[meta_df['Agreement Rate'].idxmin()]
            biggest_increase = meta_df.loc[meta_df['Grant Rate Difference'].idxmax()]
            biggest_decrease = meta_df.loc[meta_df['Grant Rate Difference'].idxmin()]
            
            st.markdown(f"""
            - **Highest Agreement**: {highest_agreement['Meta Topic']} ({highest_agreement['Agreement Rate']:.1%})
            - **Lowest Agreement**: {lowest_agreement['Meta Topic']} ({lowest_agreement['Agreement Rate']:.1%})
            - **Biggest Grant Rate Increase**: {biggest_increase['Meta Topic']} (+{biggest_increase['Grant Rate Difference']:.1%})
            - **Biggest Grant Rate Decrease**: {biggest_decrease['Meta Topic']} ({biggest_decrease['Grant Rate Difference']:.1%})
            """)

elif analysis_mode == "Topic Analysis":
    st.markdown("## 🎯 Topic Analysis")
    
    if len(filtered_df) == 0:
        st.warning("No data available with current filters.")
    else:
        # Topic selection
        available_topics = [t for t in filtered_df['topic'].unique() if t is not None]
        selected_analysis_topic = st.selectbox("Select topic for detailed analysis:", available_topics)
        
        topic_data = filtered_df[filtered_df['topic'] == selected_analysis_topic]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Decision distribution for this topic
            decision_dist = topic_data['decision_pattern'].value_counts()
            fig = px.pie(values=decision_dist.values, names=decision_dist.index,
                         title=f"Decision Distribution: {selected_analysis_topic}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Grant rates comparison
            pre_granted = (topic_data['pre_brexit_decision'] == 'granted').sum()
            pre_denied = (topic_data['pre_brexit_decision'] == 'denied').sum()
            post_granted = (topic_data['post_brexit_decision'] == 'granted').sum()
            post_denied = (topic_data['post_brexit_decision'] == 'denied').sum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Granted', x=['Pre-Brexit', 'Post-Brexit'], 
                                y=[pre_granted, post_granted], marker_color='green'))
            fig.add_trace(go.Bar(name='Denied', x=['Pre-Brexit', 'Post-Brexit'], 
                                y=[pre_denied, post_denied], marker_color='red'))
            fig.update_layout(title=f"Grant/Deny Counts: {selected_analysis_topic}", barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        # Disagreement cases
        disagreements = topic_data[topic_data['decisions_match'] == False]
        if len(disagreements) > 0:
            st.markdown(f"### 🔍 Disagreement Cases ({len(disagreements)})")
            
            # Show sample cases
            for idx, (_, row) in enumerate(disagreements.head(3).iterrows()):
                with st.expander(f"Case {row['sample_id']} - {row['decision_pattern']}"):
                    st.markdown(f"**Demographics**: {row['gender']}, {row['age']}, {row['country']}, {row['religion']}")
                    st.markdown(f"**Vignette**: {row['vignette_text'][:300]}...")

elif analysis_mode == "Protected Attributes":
    st.markdown("## 👥 Protected Attributes Analysis")
    
    attribute = st.selectbox("Select attribute:", ['gender', 'age', 'religion', 'country'])
    
    # Calculate stats by attribute
    attr_stats = []
    for value in filtered_df[attribute].unique():
        if value == 'Unknown':
            continue
        
        subset = filtered_df[filtered_df[attribute] == value]
        if len(subset) < 3:  # Skip small groups
            continue
        
        stats = {
            'Value': str(value),
            'Cases': len(subset),
            'Agreement Rate': subset['decisions_match'].mean(),
            'Pre-Brexit Grant Rate': (subset['pre_brexit_decision'] == 'granted').mean(),
            'Post-Brexit Grant Rate': (subset['post_brexit_decision'] == 'granted').mean(),
        }
        attr_stats.append(stats)
    
    if attr_stats:
        attr_df = pd.DataFrame(attr_stats).sort_values('Cases', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Grant rates by attribute
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Pre-Brexit', x=attr_df['Value'], 
                                y=attr_df['Pre-Brexit Grant Rate'], marker_color='blue'))
            fig.add_trace(go.Bar(name='Post-Brexit', x=attr_df['Value'], 
                                y=attr_df['Post-Brexit Grant Rate'], marker_color='red'))
            fig.update_layout(title=f"Grant Rates by {attribute.title()}", barmode='group')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Agreement rates by attribute
            fig = px.bar(attr_df, x='Value', y='Agreement Rate',
                         title=f"Agreement Rates by {attribute.title()}")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### 📊 Detailed Statistics")
        st.dataframe(attr_df, use_container_width=True)

elif analysis_mode == "Individual Cases":
    st.markdown("## 📄 Individual Cases")
    
    # Case filters
    col1, col2 = st.columns(2)
    with col1:
        case_filter = st.selectbox("Filter cases:", 
                                   ['All Cases', 'Disagreement Only', 'Granted→Denied', 'Denied→Granted'])
    with col2:
        search_term = st.text_input("Search in vignette:")
    
    # Apply case filters
    case_filtered_df = filtered_df.copy()
    
    if case_filter == 'Disagreement Only':
        case_filtered_df = case_filtered_df[case_filtered_df['decisions_match'] == False]
    elif case_filter == 'Granted→Denied':
        case_filtered_df = case_filtered_df[
            (case_filtered_df['pre_brexit_decision'] == 'granted') & 
            (case_filtered_df['post_brexit_decision'] == 'denied')
        ]
    elif case_filter == 'Denied→Granted':
        case_filtered_df = case_filtered_df[
            (case_filtered_df['pre_brexit_decision'] == 'denied') & 
            (case_filtered_df['post_brexit_decision'] == 'granted')
        ]
    
    if search_term:
        case_filtered_df = case_filtered_df[
            case_filtered_df['vignette_text'].str.contains(search_term, case=False, na=False)
        ]
    
    st.markdown(f"**Found {len(case_filtered_df)} cases**")
    
    # Display cases
    for idx, (_, row) in enumerate(case_filtered_df.head(10).iterrows()):
        with st.expander(f"Case {row['sample_id']} - {row['topic'][:40]}... - {row['decision_pattern']}"):
            # Case header with metadata
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                st.markdown(f"**Topic**: {row['topic']}")
                st.markdown(f"**Meta Topic**: {row['meta_topic']}")
                st.markdown(f"**Demographics**: {row['gender']}, {row['age']}, {row['country']}, {row['religion']}")
            
            with col_b:
                agreement_status = "✅ Agreement" if row['decisions_match'] else "❌ Disagreement"
                st.markdown(f"**Status**: {agreement_status}")
            
            with col_c:
                pre_color = "🟢" if row['pre_brexit_decision'] == 'granted' else "🔴"
                post_color = "🟢" if row['post_brexit_decision'] == 'granted' else "🔴"
                st.markdown(f"{pre_color} **Pre-Brexit**: {row['pre_brexit_decision'].title()}")
                st.markdown(f"{post_color} **Post-Brexit**: {row['post_brexit_decision'].title()}")
            
            # Vignette text
            st.markdown("### 📄 Case Vignette")
            st.markdown(f"*{row['vignette_text']}*")
            
            # Model reasoning comparison
            st.markdown("### 🤖 Model Reasoning Comparison")
            
            reasoning_col1, reasoning_col2 = st.columns(2)
            
            with reasoning_col1:
                st.markdown("#### 🇬🇧 Pre-Brexit Model (2013-2016)")
                decision_color = "#d4edda" if row['pre_brexit_decision'] == 'granted' else "#f8d7da"
                st.markdown(f"""
                <div style="background-color: {decision_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Decision: {row['pre_brexit_decision'].upper()}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                if row['pre_brexit_reasoning']:
                    st.markdown("**Reasoning:**")
                    st.markdown(f"*{row['pre_brexit_reasoning']}*")
                else:
                    st.markdown("*No reasoning available*")
            
            with reasoning_col2:
                st.markdown("#### 🌍 Post-Brexit Model (2019-2025)")
                decision_color = "#d4edda" if row['post_brexit_decision'] == 'granted' else "#f8d7da"
                st.markdown(f"""
                <div style="background-color: {decision_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Decision: {row['post_brexit_decision'].upper()}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                if row['post_brexit_reasoning']:
                    st.markdown("**Reasoning:**")
                    st.markdown(f"*{row['post_brexit_reasoning']}*")
                else:
                    st.markdown("*No reasoning available*")
            
            # Reasoning comparison insights for disagreement cases
            if not row['decisions_match']:
                st.markdown("### 🔍 Disagreement Analysis")
                if row['pre_brexit_reasoning'] and row['post_brexit_reasoning']:
                    st.markdown("**Key Differences in Reasoning:**")
                    
                    # Simple keyword analysis
                    pre_words = set(row['pre_brexit_reasoning'].lower().split())
                    post_words = set(row['post_brexit_reasoning'].lower().split())
                    
                    unique_pre = pre_words - post_words
                    unique_post = post_words - pre_words
                    
                    if unique_pre or unique_post:
                        col_diff1, col_diff2 = st.columns(2)
                        with col_diff1:
                            if unique_pre:
                                st.markdown("**Pre-Brexit unique concepts:**")
                                st.markdown(", ".join(list(unique_pre)[:10]))
                        with col_diff2:
                            if unique_post:
                                st.markdown("**Post-Brexit unique concepts:**")
                                st.markdown(", ".join(list(unique_post)[:10]))

elif analysis_mode == "Generate Report":
    st.markdown("## 📊 Automated Insights Report")
    
    if len(filtered_df) == 0:
        st.warning("No data available with current filters.")
    else:
        st.markdown("### 🔄 Generating Comprehensive Analysis...")
        
        with st.spinner("Analyzing patterns and generating insights..."):
            # Initialize insights
            insights = []
            
            # 1. Overall model behavior analysis
            total_cases = len(filtered_df)
            agreement_rate = filtered_df['decisions_match'].mean()
            pre_grant_rate = (filtered_df['pre_brexit_decision'] == 'granted').mean()
            post_grant_rate = (filtered_df['post_brexit_decision'] == 'granted').mean()
            grant_rate_change = post_grant_rate - pre_grant_rate
            
            insights.append(f"**Overall Model Behavior**: Analyzed {total_cases} cases with {agreement_rate:.1%} agreement rate.")
            
            if grant_rate_change > 0.05:
                insights.append(f"🔴 **Significant Increase**: Post-Brexit model shows {grant_rate_change:.1%} higher grant rate than Pre-Brexit model.")
            elif grant_rate_change < -0.05:
                insights.append(f"🔵 **Significant Decrease**: Post-Brexit model shows {abs(grant_rate_change):.1%} lower grant rate than Pre-Brexit model.")
            else:
                insights.append(f"⚖️ **Stable Behavior**: Similar grant rates between models (difference: {grant_rate_change:+.1%}).")
            
            # 2. Meta topic analysis
            meta_topic_insights = []
            for meta_topic in filtered_df['meta_topic'].unique():
                if meta_topic is None:
                    continue
                
                meta_data = filtered_df[filtered_df['meta_topic'] == meta_topic]
                if len(meta_data) < 10:  # Skip small groups
                    continue
                
                meta_pre_grant = (meta_data['pre_brexit_decision'] == 'granted').mean()
                meta_post_grant = (meta_data['post_brexit_decision'] == 'granted').mean()
                meta_change = meta_post_grant - meta_pre_grant
                meta_agreement = meta_data['decisions_match'].mean()
                
                if abs(meta_change) > 0.1:  # Significant change
                    direction = "more inclined to grant" if meta_change > 0 else "less inclined to grant"
                    meta_topic_insights.append(f"📈 **{meta_topic}**: Post-Brexit model is {direction} cases ({meta_change:+.1%} change, {meta_agreement:.1%} agreement).")
            
            insights.extend(meta_topic_insights)
            
            # 3. Protected attributes analysis
            protected_attrs = ['gender', 'religion', 'country', 'age']
            
            for attr in protected_attrs:
                attr_insights = []
                attr_values = filtered_df[attr].value_counts()
                
                # Only analyze values with sufficient data
                for value in attr_values.index[:5]:  # Top 5 values
                    if attr_values[value] < 10:
                        continue
                    
                    subset = filtered_df[filtered_df[attr] == value]
                    pre_grant = (subset['pre_brexit_decision'] == 'granted').mean()
                    post_grant = (subset['post_brexit_decision'] == 'granted').mean()
                    change = post_grant - pre_grant
                    
                    if abs(change) > 0.15:  # Significant change
                        direction = "significantly more favorable" if change > 0 else "significantly less favorable"
                        attr_insights.append(f"👥 **{attr.title()} - {value}**: Post-Brexit model is {direction} ({change:+.1%}, n={len(subset)}).")
                
                insights.extend(attr_insights)
            
            # 4. Topic-specific analysis
            topic_insights = []
            for topic in filtered_df['topic'].value_counts().head(10).index:
                topic_data = filtered_df[filtered_df['topic'] == topic]
                if len(topic_data) < 5:
                    continue
                
                topic_pre_grant = (topic_data['pre_brexit_decision'] == 'granted').mean()
                topic_post_grant = (topic_data['post_brexit_decision'] == 'granted').mean()
                topic_change = topic_post_grant - topic_pre_grant
                topic_agreement = topic_data['decisions_match'].mean()
                
                if abs(topic_change) > 0.2:  # Very significant change
                    direction = "much more likely to grant" if topic_change > 0 else "much less likely to grant"
                    topic_insights.append(f"🎯 **{topic}**: Post-Brexit model is {direction} cases ({topic_change:+.1%}, {topic_agreement:.1%} agreement, n={len(topic_data)}).")
            
            insights.extend(topic_insights)
            
            # 5. Cross-sectional insights (topic + attribute combinations)
            cross_insights = []
            
            # Analyze gender differences within specific topics
            for topic in filtered_df['topic'].value_counts().head(5).index:
                topic_data = filtered_df[filtered_df['topic'] == topic]
                
                for gender in ['Male', 'Female']:
                    gender_topic_data = topic_data[topic_data['gender'] == gender]
                    if len(gender_topic_data) < 5:
                        continue
                    
                    pre_grant = (gender_topic_data['pre_brexit_decision'] == 'granted').mean()
                    post_grant = (gender_topic_data['post_brexit_decision'] == 'granted').mean()
                    change = post_grant - pre_grant
                    
                    if abs(change) > 0.25:  # Very significant change
                        direction = "much more favorable to" if change > 0 else "much less favorable to"
                        cross_insights.append(f"⚖️ **Gender Bias in {topic}**: Post-Brexit model is {direction} {gender.lower()} applicants ({change:+.1%}, n={len(gender_topic_data)}).")
            
            insights.extend(cross_insights)
            
            # 6. Disagreement patterns
            disagreements = filtered_df[~filtered_df['decisions_match']]
            if len(disagreements) > 0:
                # Most disagreed topics
                disagreement_topics = disagreements['topic'].value_counts().head(3)
                insights.append(f"🔍 **High Disagreement Topics**: {', '.join(disagreement_topics.index)} show the most disagreements between models.")
                
                # Disagreement by protected attributes
                for attr in ['gender', 'religion']:
                    attr_disagreements = disagreements[attr].value_counts()
                    if len(attr_disagreements) > 0:
                        top_disagreement = attr_disagreements.index[0]
                        disagreement_rate = len(disagreements[disagreements[attr] == top_disagreement]) / len(filtered_df[filtered_df[attr] == top_disagreement])
                        if disagreement_rate > 0.3:  # High disagreement rate
                            insights.append(f"⚠️ **High Disagreement Group**: {attr.title()} '{top_disagreement}' shows {disagreement_rate:.1%} disagreement rate.")
        
        # Display insights
        st.markdown("### 📋 Generated Insights")
        
        if insights:
            for i, insight in enumerate(insights, 1):
                st.markdown(f"{i}. {insight}")
        else:
            st.markdown("No significant patterns detected with current filters.")
        
        # Summary statistics table
        st.markdown("### 📊 Summary Statistics")
        
        summary_stats = {
            'Metric': [
                'Total Cases',
                'Agreement Rate',
                'Pre-Brexit Grant Rate',
                'Post-Brexit Grant Rate',
                'Grant Rate Change',
                'Disagreement Cases',
                'Unique Topics',
                'Unique Meta Topics'
            ],
            'Value': [
                f"{total_cases:,}",
                f"{agreement_rate:.1%}",
                f"{pre_grant_rate:.1%}",
                f"{post_grant_rate:.1%}",
                f"{grant_rate_change:+.1%}",
                f"{len(disagreements):,}",
                f"{filtered_df['topic'].nunique()}",
                f"{filtered_df['meta_topic'].nunique()}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download report button
        if st.button("📥 Download Report"):
            report_text = f"""
# AI Model Evaluation Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Cases Analyzed: {total_cases:,}
- Agreement Rate: {agreement_rate:.1%}
- Pre-Brexit Grant Rate: {pre_grant_rate:.1%}
- Post-Brexit Grant Rate: {post_grant_rate:.1%}
- Grant Rate Change: {grant_rate_change:+.1%}

## Key Insights
"""
            for i, insight in enumerate(insights, 1):
                # Remove markdown formatting for plain text
                clean_insight = insight.replace('**', '').replace('🔴', '').replace('🔵', '').replace('⚖️', '').replace('📈', '').replace('👥', '').replace('🎯', '').replace('⚠️', '').replace('🔍', '')
                report_text += f"{i}. {clean_insight}\n"
            
            st.download_button(
                label="📄 Download as Text Report",
                data=report_text,
                file_name=f"ai_model_evaluation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

else:  # Cross-sectional Analysis
    st.markdown("## 🔀 Cross-sectional Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        cross_topic = st.selectbox("Topic:", [t for t in filtered_df['topic'].unique() if t is not None])
    with col2:
        cross_attr = st.selectbox("Attribute:", ['gender', 'age', 'religion', 'country'])
    
    # Cross-sectional analysis
    cross_data = filtered_df[filtered_df['topic'] == cross_topic]
    
    if len(cross_data) > 0:
        cross_stats = []
        for attr_value in cross_data[cross_attr].unique():
            if attr_value == 'Unknown':
                continue
            
            subset = cross_data[cross_data[cross_attr] == attr_value]
            if len(subset) < 3:
                continue
            
            stats = {
                f'{cross_attr.title()}': str(attr_value),
                'Cases': len(subset),
                'Pre-Brexit Grant Rate': (subset['pre_brexit_decision'] == 'granted').mean(),
                'Post-Brexit Grant Rate': (subset['post_brexit_decision'] == 'granted').mean(),
                'Agreement Rate': subset['decisions_match'].mean(),
            }
            cross_stats.append(stats)
        
        if cross_stats:
            cross_df = pd.DataFrame(cross_stats)
            
            # Visualization
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Grant Rates', 'Agreement Rates'))
            
            fig.add_trace(
                go.Bar(x=cross_df[f'{cross_attr.title()}'], y=cross_df['Pre-Brexit Grant Rate'], 
                      name='Pre-Brexit Grant', marker_color='blue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=cross_df[f'{cross_attr.title()}'], y=cross_df['Post-Brexit Grant Rate'], 
                      name='Post-Brexit Grant', marker_color='red'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=cross_df[f'{cross_attr.title()}'], y=cross_df['Agreement Rate'], 
                      name='Agreement Rate', marker_color='green'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, 
                            title=f"{cross_topic} by {cross_attr.title()}")
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(cross_df, use_container_width=True)
        else:
            st.warning("Not enough data for cross-sectional analysis with current filters.")
    else:
        st.warning("No data available for selected topic.") 