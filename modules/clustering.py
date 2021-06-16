import numpy as np

from sklearn.cluster import MiniBatchKMeans as mbk

from tslearn.clustering import TimeSeriesKMeans as tskm
from .visualization import visualize_auto_elbow


def rule_based_clustering(df, feature='nth_day'):
    """
    """
    df = df.sort_values(['account_id', 'nth_match'])
    feature_values = df.groupby('account_id')[feature].agg('max').values
    labels = []
    for value in feature_values:

        if value <= 25:
            labels.append(0)
        elif 76 <= value <= 90:
            labels.append(1)
        elif 136 <= value <= 150:
            labels.append(2)
        else:
            labels.append(-1)

    return np.array(labels)


def auto_elbow(n_clusters, inertias, verbose=True,
               save_path='results\\figures\\base'):
    """
    """
    y = (inertias[0], inertias[-1])
    x = (n_clusters[0], n_clusters[-1])

    alpha, beta = np.polyfit(
        x,
        y,
        1
    )
    grad_line = [beta+(alpha*k) for k in n_clusters]
    optimal_k = np.argmax([grad - i for grad, i in zip(grad_line, inertias)])
    optimal_k = n_clusters[optimal_k]
    if verbose:
        print(f'Optimal K found at {optimal_k}')
        visualize_auto_elbow(
            n_clusters=n_clusters,
            inertias=inertias,
            grad_line=grad_line,
            optimal_k=optimal_k,
            save_path=save_path
        )
    return optimal_k


def auto_k_means(X, min_k=2, max_k=15, save_path='results\\figures\\base',
                 **kwargs):
    """
    """
    inertias = []
    n_centroids = [i for i in range(min_k, max_k)]

    for k in n_centroids:

        print(f'Clustering {k}')
        if len(X.shape) == 3:
            clust = tskm(
                n_clusters=k,
                n_jobs=-1,
                verbose=0,
                **kwargs
            )
        else:
            clust = mbk(
                n_clusters=k,
                **kwargs
            )
        clust.fit(X)
        inertias.append(clust.inertia_)

    optimal_k = auto_elbow(
        n_clusters=[i for i in range(min_k, max_k)],
        inertias=inertias,
        verbose=True,
        save_path=save_path
    )
    if len(X.shape) == 3:
        clust = tskm(
            n_clusters=optimal_k,
            n_jobs=-1,
            verbose=0,
            **kwargs
        )
    else:
        clust = mbk(
            n_clusters=optimal_k,
            **kwargs
        )
    clust.fit(X)
    return clust
