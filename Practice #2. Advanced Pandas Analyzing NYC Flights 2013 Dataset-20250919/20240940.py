import pandas as pd
import numpy as np

# WARNING: Only modify code within the "WRITE YOUR CODE" blocks below.
"""
IMPORTANT NOTES
    - Edit Scope: Modify only inside WRITE YOUR CODE blocks; any change outside (including comments/spacing) may cause errors and point deductions.
    - Required Vars: Each block already includes grading-required variables; do not rename/remove them or change types/return formats.
    - Grading Rule: Evaluation is by exact match of print output—whitespace, newlines, case, and punctuation must be identical. Do not add extra prints anywhere.
    - Before Submit: Confirm no edits were made outside the designated blocks.
"""



# Task 1: Load and clean
df = pd.read_csv('nycflights.csv')  # Or local path

### WRITE YOUR CODE BELOW
df['dep_time_hour'] = (df['dep_time']//100) + (df['dep_time']%100)/60
df['arr_time_hour'] = (df['arr_time']//100) + (df['arr_time']%100)/60
df_clean = df.loc[~df['arr_delay'].isna()].reset_index(drop=True)
df_clean['carrier'] = df_clean['carrier'].astype('category')
df_clean['origin'] = df_clean['origin'].astype('category')
### WRITE YOUR CODE ABOVE
print('Shape:', df_clean.shape)
print('Memory (MB):', df_clean.memory_usage(deep=True).sum() / 1e6)
print(df_clean.dtypes)



# Task 2: Outliers

### WRITE YOUR CODE BELOW
# TODO: IQR mask
Q1 = df_clean['arr_delay'].quantile(0.25)
Q3 = df_clean['arr_delay'].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

df_filtered = df_clean.loc[(df_clean['arr_delay'] >= lower) & (df_clean['arr_delay'] <= upper)].copy()
outliers_removed = len(df_clean) - len(df_filtered)
### WRITE YOUR CODE ABOVE
print('Outliers removed:', outliers_removed)

### WRITE YOUR CODE BELOW
# TODO: delay_category with pd.cut
df_filtered['delay_category'] = pd.cut(df_filtered['arr_delay'], bins=[-np.inf, 15, 60, np.inf], labels=['short', 'medium', 'long'])
### WRITE YOUR CODE ABOVE
print(df_filtered['delay_category'].value_counts())



# Task 3: Groupby agg
### WRITE YOUR CODE BELOW
agg_dict = {'arr_delay' : ['mean', lambda x: (x>60).mean()], 
            'air_time' : 'median',
            'distance' : lambda x: np.percentile(x, 95)
            }
grouped_temp = df_filtered.groupby(['origin', 'carrier'], observed = True).agg(agg_dict)
grouped_temp.columns = ['_'.join([c for c in tpl if c]).strip() for tpl in grouped_temp.columns.values]
grouped_temp = grouped_temp.rename(columns = {
    'arr_delay_mean': 'mean_arr_delay',
    'arr_delay_<lambda_0>': 'prop_long_delay',
    'air_time_median': 'median_air_time',
    'distance_<lambda>': 'p95_distance'
})
grouped = grouped_temp.sort_values('mean_arr_delay', ascending=False).head(5)
grouped = grouped.reset_index()
### WRITE YOUR CODE ABOVE
print(grouped.head())



# Task 4: Pivot and melt
### WRITE YOUR CODE BELOW
pivot_table = df_filtered.pivot_table(
    index = 'month',
    columns = 'carrier',
    values = 'dep_delay',
    aggfunc = 'mean',
    fill_value = 0,
    margins = True,
    observed = False
)
melted = pivot_table.reset_index().melt(
    id_vars = 'month',
    var_name = 'carrier',
    value_name = 'mean_dep_delay'
)

melted['metric'] = 'mean_dep_delay'
top_carriers = df_filtered['carrier'].value_counts().nlargest(3).index
melted = melted[melted['carrier'].isin(top_carriers)]
melted = melted[melted['month'] != 'All']
corr = melted['mean_dep_delay'].corr(melted['month'])
### WRITE YOUR CODE ABOVE
print('Shape:', melted.shape, 'Corr:', round(corr, 2))



# Task 5: Windows
### WRITE YOUR CODE BELOW
mo_avg = df_filtered.groupby(['month', 'origin', 'carrier'], observed = True)['arr_delay'].mean().reset_index(name = 'mean_arr_delay')
mo_avg['dense_rank'] = mo_avg.groupby(['month', 'origin'], observed = True)['mean_arr_delay'].rank(method = 'dense', ascending = True)

carrier_month = df_filtered.groupby(['carrier', 'month'], observed = True)['arr_delay'].mean().reset_index().sort_values(['carrier', 'month'])
carrier_month['cumulative_avg'] = carrier_month.groupby('carrier', observed = True)['arr_delay'].expanding().mean().reset_index(level=0, drop=True)

worst = mo_avg.loc[mo_avg.groupby('month')['mean_arr_delay'].idxmax()].set_index('month')['carrier'].sort_index()
top3 = carrier_month.groupby('carrier', observed = True)['cumulative_avg'].last().sort_values(ascending = False).head(3)
### WRITE YOUR CODE ABOVE
print(worst)
print(top3)


# Task 6: Filter and score
### WRITE YOUR CODE BELOW
airT_90 = df_filtered['air_time'].quantile(0.9)
df_filtered_task6 = df_filtered[(df_filtered['distance']>1000) & (df_filtered['air_time']>airT_90)]
df_filtered_task6 = df_filtered_task6.assign(score=(df_filtered_task6['dep_delay'] + df_filtered_task6['arr_delay']) / df_filtered_task6['distance'] * 100)

grouped_by_dest = df_filtered_task6.groupby('dest').agg(mean_score = ('score', 'mean'), count = ('dest', 'count'))
worst_dests = grouped_by_dest[grouped_by_dest['count'] > 1000].sort_values('mean_score', ascending = False).head(5)
worst_dests = worst_dests.reset_index().to_string(index=False)
### WRITE YOUR CODE ABOVE
print(worst_dests)