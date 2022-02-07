import pandas as pd
import numpy as np

from tqdm import tqdm 

tqdm.pandas()


def compute_means(df, var, n):
    """Return an array containing the mean of variable 'var' at each match for all account_ids in dataframe 'df' over 'n' matches"""
    matches = df.groupby('account_id', sort=False).head(n)
    means = matches.groupby('nth_match')[var].mean().tolist()

    return means


def compute_errors(df, var, n):
    """Return an array containing the standard error of variable 'var' at each match for all account_ids in dataframe 'df' over 'n' matches"""
    matches = df.groupby('account_id', sort=False).head(n)
    errors = matches.groupby('nth_match')[var].sem()
    errors = errors.tolist()

    return errors


def compute_stds(df, var, n):
    """Return an array containing the standard deviation of variable 'var' at each match for all account_ids in dataframe 'df' over 'n' matches"""
    matches = df.groupby('account_id', sort=False).head(n)
    stds = matches.groupby('nth_match')[var].std()
    stds = stds.tolist()
    
    return stds


def minmax_scaler(df, var):
    """Standardise a column 'var' in dataframe 'df' and return standardised column as array"""
    v_max = df[var].max()
    v_min = df[var].min()
    scores = (df[var] - v_min) / (v_max - v_min)
    
    return scores


def mean_of_first_n_matches(df, var, n):
    """Computes for each player the mean of variable var for the first n matches and returns means as a list"""
    players = df.groupby('account_id', sort=False)
    means = players.apply(lambda x: x[var].head(n).mean()).tolist()
    
    return means


def create_summary_dataframe(df, dvs, n_matches, head=5, tail=5, spacing='time_delta', verbose=True):
    """Returns an abbreviated dataframe consisting of player ids with corresponding acqusition score on specified dependent variable, initial performance, and specified spacing metric"""
    account_id = df['account_id'].unique().tolist()
    # spacing is defined as the time difference between the last and first game
    if spacing == 'time_delta':
        nth_match_time = df.groupby('account_id', sort=False).nth(n_matches - 1)['time_stamp']
        first_match_time = df.groupby('account_id', sort=False).nth(0)['time_stamp']
        spacing = ((nth_match_time - first_match_time) / 86400000).tolist()
        
    # create dataframe
    data = pd.DataFrame({'account_id': account_id,
                         'spacing': spacing,})
    # create summary stats of performance variables
    head = df.groupby('account_id').head(5)
    tail = df.groupby('account_id').tail(5)
    for dv in dvs:
        # acquisition measure is difference in dv between mean of last 5 games and first 5 games
        head_average = head.groupby('account_id', sort=False)[dv].mean()
        tail_average = tail.groupby('account_id', sort=False)[dv].mean()

        acquisition = tail_average - head_average
        
        data[f'{dv}_initial_perf'] = head_average.tolist()
        data[f'{dv}_acquisition'] = acquisition.tolist()

    if verbose == True:
        display(data.head())
    
    return data


def games_per_time_period(df, days_per_period=7):
    """Return list of describing n matches played each time period in days, from start to end of dataframe"""
    # define start and end dates to observe
    start = df['date'].min()
    # end is specified as 6 days after recorded final date of play, so pd.date_range behaves as we want for players with under 7 days of play 
    end = pd.to_datetime(df['date'].max())
    end = end.strftime('%Y-%m-%d')  # back to string
    
    # list end dates of each time period starting from first recorded day
    dates = [date.strftime('%Y-%m-%d')  # format all pandas datetime values as string
                 for date
                 in pd.date_range(start=start, end=end)]  # periodise between dates by specified unit

    # get match counts per time period
    counts = []
    periods = []
    while len(dates) > 0:
        start = dates[0]
        end = dates[days_per_period - 1] if len(dates) >= days_per_period else dates[len(dates) - 1]
        period = df[(df['date'] >= start) & (df['date'] <= end)]
        count = len(period)
        periods.append((start, end))
        counts.append(count)
        del dates[:days_per_period]
        
    return counts
