from tqdm import tqdm

import pandas as pd

TARGETS = [
    'time_gap',
    'rating',
    'gpm',
    'kda',
    'kdr',
    'win'
]

df = pd.read_csv(
    'data\\df_100_100.csv',
    usecols=[
        'account_id',
        'nth_match',
        'nth_day',
        'date',
        'time_gap',
        'rating',
        'gpm',
        'kda',
        'kdr',
        'win',
        'position',
        'region'
    ]
)

df = df.sort_values(['account_id', 'nth_match'])
df.to_csv('data\\df_non_smoothed.csv', index=False)

# MOVING AVERAGE FOR SMOOTHING (EXPONENTIAL MOVING AVERAGE WINDOW 10)
for target in tqdm(TARGETS):

    # we run and exponential moving average over a window of 10
    df[target] = df.groupby('account_id')[target].apply(
        lambda x: x.ewm(span=10).mean()
    )

df = df.sort_values(['account_id', 'nth_match'])
df.to_csv('data\\df_smoothed.csv', index=False)
