import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the CSV file
data_path = '/data/shil6369/gov_scraper/preprocessing/data/processed_data/processed_legal_cases.csv'
print("Loading data...")
df = pd.read_csv(data_path, low_memory=False)

print(f"Dataset shape: {df.shape}")
print(f"Total cases: {len(df):,}")

# Print all column names
print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

# Print nulls in the main text column
text_col = 'decision_text_cleaned'
if text_col in df.columns:
    nulls = df[text_col].isnull().sum()
    print(f"\nNulls in '{text_col}': {nulls} ({100*nulls/len(df):.2f}%)")
else:
    print(f"\nColumn '{text_col}' not found!")

# Print unique values for all columns containing 'year' in their name
year_cols = [col for col in df.columns if 'year' in col.lower()]
print("\n=== YEAR COLUMNS AND SAMPLE VALUES ===")
for col in year_cols:
    print(f"{col}: nulls={df[col].isnull().sum()}, unique (non-null) sample={df[col].dropna().unique()[:10]}")

# Focus on the key columns for analysis
print("\n=== DATA OVERVIEW ===")
print(f"Cases with text length data: {df['decision_text_length'].notna().sum():,}")
print(f"Cases with promulgation date data: {df['promulgation_date_standardized'].notna().sum():,}")

# Extract year from promulgation_date_standardized
df['extracted_year'] = pd.to_datetime(df['promulgation_date_standardized']).dt.year

# Write the extracted year back to the case_year column
df['case_year'] = df['extracted_year']

# Filter data to only include cases with both year and text length, and from 2013 onwards
analysis_df = df.dropna(subset=['extracted_year', 'decision_text_length']).copy()
analysis_df = analysis_df[analysis_df['extracted_year'] >= 2013]

print(f"\nCases available for analysis (2013 onwards): {len(analysis_df):,}")
print(f"Year range: {analysis_df['extracted_year'].min()} to {analysis_df['extracted_year'].max()}")

# Calculate statistics by year
yearly_stats = analysis_df.groupby('extracted_year').agg({
    'decision_text_length': ['count', 'mean', 'median', 'std', 'min', 'max']
}).round(2)

# Flatten column names
yearly_stats.columns = ['case_count', 'avg_length', 'median_length', 'std_length', 'min_length', 'max_length']
yearly_stats = yearly_stats.reset_index()

print("\n=== YEARLY STATISTICS (2013 onwards) ===")
print(yearly_stats.to_string(index=False))

# Create the main plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plot 1: Average case length by year
ax1.plot(yearly_stats['extracted_year'], yearly_stats['avg_length'], 
         marker='o', linewidth=2, markersize=8, color='steelblue')
ax1.fill_between(yearly_stats['extracted_year'], 
                 yearly_stats['avg_length'] - yearly_stats['std_length'],
                 yearly_stats['avg_length'] + yearly_stats['std_length'],
                 alpha=0.3, color='steelblue', label='±1 Standard Deviation')

ax1.set_title('Average Case Text Length by Year (2013 onwards)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Average Text Length (characters)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add value labels on points
for i, row in yearly_stats.iterrows():
    ax1.annotate(f'{row["avg_length"]:,.0f}', 
                (row['extracted_year'], row['avg_length']), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center', 
                fontsize=9)

# Plot 2: Number of cases by year
ax2.bar(yearly_stats['extracted_year'], yearly_stats['case_count'], 
        color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
ax2.set_title('Number of Cases by Year (2013 onwards)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Number of Cases', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, row in yearly_stats.iterrows():
    ax2.annotate(f'{row["case_count"]:,}', 
                (row['extracted_year'], row['case_count']), 
                textcoords="offset points", 
                xytext=(0,5), 
                ha='center', 
                fontsize=9)

plt.tight_layout()
plt.savefig('case_length_analysis_2013_onwards.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Summary statistics
print("\n=== OVERALL STATISTICS (2013 onwards) ===")
print(f"Overall average case length: {analysis_df['decision_text_length'].mean():,.0f} characters")
print(f"Overall median case length: {analysis_df['decision_text_length'].median():,.0f} characters")
print(f"Overall standard deviation: {analysis_df['decision_text_length'].std():,.0f} characters")

# Find years with highest and lowest average lengths
max_avg_year = yearly_stats.loc[yearly_stats['avg_length'].idxmax()]
min_avg_year = yearly_stats.loc[yearly_stats['avg_length'].idxmin()]

print(f"\nYear with highest average case length: {int(max_avg_year['extracted_year'])} "
      f"({max_avg_year['avg_length']:,.0f} characters, {max_avg_year['case_count']} cases)")
print(f"Year with lowest average case length: {int(min_avg_year['extracted_year'])} "
      f"({min_avg_year['avg_length']:,.0f} characters, {min_avg_year['case_count']} cases)")

# Trend analysis
if len(yearly_stats) > 1:
    correlation = np.corrcoef(yearly_stats['extracted_year'], yearly_stats['avg_length'])[0, 1]
    print(f"\nCorrelation between year and average case length: {correlation:.3f}")
    
    if correlation > 0.1:
        print("Trend: Increasing case length over time")
    elif correlation < -0.1:
        print("Trend: Decreasing case length over time")
    else:
        print("Trend: No clear trend in case length over time")

print(f"\nPlot saved as 'case_length_analysis_2013_onwards.png'")

# Save the updated dataframe with corrected case_year
print(f"\nSaving updated dataset with corrected case_year column...")
df.to_csv('/data/shil6369/gov_scraper/preprocessing/data/processed_data/processed_legal_cases_updated.csv', index=False)
print("Updated dataset saved as 'processed_legal_cases_updated.csv'") 