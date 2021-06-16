import numpy as np

import pandas as pd

from tslearn.preprocessing import TimeSeriesScalerMeanVariance as tsmv

from umap import UMAP


for modality in ['non_smoothed']:

    df = pd.read_csv(f'data\\df_{modality}.csv')
    df = df.sort_values(['account_id', 'nth_match'])

    unique_ids = len(df['account_id'].unique())
    X = df['time_gap'].values
    X = X.reshape((unique_ids, 100))
    X = X[:, 1:95]
    X = tsmv().fit_transform(X)

    for n_components, name in zip([2, 20], ['viz', 'feat']):

        embedding = UMAP(
            n_components=n_components,
            n_neighbors=100,
            verbose=True,
            n_epochs=1000,
            metric='euclidean'
        ).fit_transform(X.reshape(X.shape[0], X.shape[1]))

        np.save(
            f'results\\arrays\\embedding_{name}_{modality}',
            embedding
        )
