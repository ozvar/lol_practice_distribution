from sklearn.preprocessing import KBinsDiscretizer

def create_filters(df, targets, cal_period=10, n_bins=3, to_retain_bins=[1]):
    """Individuate account_id to retain for each target metric 
    given the bin in which the median perfromance in the first 5 
    matches for that specific target
    """
    filter_df = df.groupby('account_id').head(cal_period)
    filter_df = filter_df.groupby('account_id')[targets].median()
    filter_df[targets] = KBinsDiscretizer(
        n_bins=n_bins, 
        encode='ordinal'
    ).fit_transform(filter_df[targets].values)
    filter_df = filter_df.reset_index()
    
    accounts_to_retain = {
    target: filter_df[filter_df[target].isin(to_retain_bins)]['account_id'].values for target in targets
    }
    return accounts_to_retain