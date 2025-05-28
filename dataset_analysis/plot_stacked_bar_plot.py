import pandas as pd
import matplotlib.pyplot as plt

# Load your detailed records (if you already have a CSV from previous analysis)
df = pd.read_csv('determination_types_detailed_by_period.csv')

# Ensure year is integer and drop rows with missing years
df = df.dropna(subset=['year'])
df['year'] = df['year'].astype(int)

# Group by year and determination_type, then count
pivot = df.pivot_table(index='year', columns='determination_type', aggfunc='size', fill_value=0)

# Sort by year
pivot = pivot.sort_index()

# Plot
pivot.plot(kind='bar', stacked=True, figsize=(18, 8), colormap='tab20')
plt.title('Number of Cases per Year by Determination Type')
plt.xlabel('Year')
plt.ylabel('Number of Cases')
plt.legend(title='Determination Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('cases_per_year_by_type.png')
plt.show()
