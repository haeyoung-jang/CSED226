import pandas as pd
import numpy as np

# Task 1
df = pd.read_csv('nycflights.csv')

def time_to_hours(time_col):
    time_str = time_col.astype(str).str.zfill(4)
    return pd.to_timedelta(time_str + ' minutes').dt.total_seconds() / 3600

df['dep_time_hours'] = time_to_hours(df['dep_time'])
df['arr_time_hours'] = time_to_hours(df['arr_time'])

df['carrier'] = df['carrier'].astype('category')
df['origin'] = df['origin'].astype('category')

df_clean = df.dropna(subset=['arr_delay']).reset_index(drop=True)

print('Shape:', df_clean.shape)
print('Memory (MB):', df_clean.memory_usage(deep=True).sum() / 1e6)
print(df_clean.dtypes)

# Task 2
arr_delay = df_clean['arr_delay']
Q1 = arr_delay.quantile(0.25)
Q3 = arr_delay.quantile(0.75)
IQR = Q3 - Q1
mask = (arr_delay >= (Q1 - 1.5 * IQR)) & (arr_delay <= (Q3 + 1.5 * IQR))
df_filtered = df_clean[mask].copy()

outliers_removed = len(df_clean) - len(df_filtered)
print('Outliers removed:', outliers_removed)

bins = [-np.inf, 15, 60, np.inf]
labels = ['short', 'medium', 'long']
df_filtered['delay_category'] = pd.cut(df_filtered['arr_delay'], bins=bins, labels=labels)
print(df_filtered['delay_category'].value_counts())

# Task 3
agg_dict = {
    'arr_delay': ['mean', lambda x: (x > 60).mean()],
    'air_time': 'median',
    'distance': lambda x: np.percentile(x, 95)
}
grouped = df_filtered.groupby(['origin', 'carrier']).agg(agg_dict).round(2)

grouped.columns = ['mean_arr_delay', 'prop_long_delay', 'median_air_time', 'p95_distance']
grouped = grouped.sort_values('mean_arr_delay', ascending=False)
print(grouped.head())

# Task 4
pivot = pd.pivot_table(
    df_filtered,
    values="dep_delay",
    index="month",
    columns="carrier",
    aggfunc="mean",
    fill_value=0,
    observed=True,
)
top_carriers = df_filtered["carrier"].value_counts().nlargest(3).index
pivot_top = pivot[top_carriers].round(1)

melted = pd.melt(
    pivot_top.reset_index(),
    id_vars=["month"],
    var_name="carrier",
    value_name="mean_dep_delay",
)
corr = melted["mean_dep_delay"].corr(melted["month"])
print('Shape:', melted.shape)
print('Corr:', round(corr, 2))

# Task 5
moc = (
    df_filtered.groupby(["month", "origin", "carrier"], observed=True)["arr_delay"]
    .mean()
    .reset_index(name="mean_arr_delay")
)
moc["rank_delay"] = moc.groupby(["month", "origin"], observed=True)[
    "mean_arr_delay"
].rank(method="dense", ascending=True)

df_filtered = df_filtered.merge(
    moc[["month", "origin", "carrier", "rank_delay"]],
    on=["month", "origin", "carrier"],
    how="left",
)

mc = (
    df_filtered.groupby(["carrier", "month"], observed=True)["arr_delay"]
    .mean()
    .reset_index(name="mean_arr_delay")
)
mc = mc.sort_values(["carrier", "month"])
mc["cum_avg_delay"] = (
    mc.groupby("carrier", observed=True)["mean_arr_delay"]
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

monthly_means = df_filtered.groupby(["month", "carrier"], observed=True)["arr_delay"].mean()
worst = (
    monthly_means.groupby("month", observed=True).idxmax().map(lambda idx: idx[1])
)
top3 = (
    mc.groupby("carrier", observed=True)["cum_avg_delay"]
    .last()
    .nlargest(3)
    .round(1)
)
print(worst)
print(top3)

# Task 6
p90_air = df_filtered["air_time"].quantile(0.9)
long_flights = df_filtered[(df_filtered["distance"] > 1000) & (df_filtered["air_time"] > p90_air)].copy()
long_flights["score"] = (
    (long_flights["dep_delay"] + long_flights["arr_delay"])
    / long_flights["distance"]
    * 100
)
grouped_dest = (
    long_flights.groupby("dest", observed=True)
    .agg(mean_score=("score", "mean"), count=("flight", "count"))
    .round(2)
)
worst_dests = (
    grouped_dest[grouped_dest["count"] > 1000]
    .sort_values("mean_score", ascending=False)
    .head(5)
)
print(worst_dests)
