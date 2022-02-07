import numpy as np
import pymc3 as pm

from sklearn.cluster import MiniBatchKMeans as mbk

from tslearn.clustering import TimeSeriesKMeans as tskm
from tslearn.preprocessing import TimeSeriesScalerMeanVariance as tsmv

from hdbscan import HDBSCAN
from umap import UMAP

from tensorflow.keras.callbacks import EarlyStopping

from visualization import visualize_auto_elbow


def auto_elbow(n_clusters, inertias, n_matches, start_gap, end_gap, fig_dir, verbose=True):
    """Calculate and return optimal k number of clusters based on passed inertias from n_clusters clusters"""
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
        print('Optimal k found at {}'.format(optimal_k))
        visualize_auto_elbow(
            n_clusters=n_clusters,
            inertias=inertias,
            n_matches=n_matches,
            start_gap=start_gap,
            end_gap=end_gap,
            fig_dir=fig_dir,
            grad_line=grad_line,
            optimal_k=optimal_k
        )
        
    return optimal_k, grad_line


def auto_k_means(X, min_k=2, max_k=15, n_matches=100, start_gap=1, end_gap=5, fig_dir='figures',
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
        n_matches=n_matches,
        start_gap=start_gap,
        end_gap=end_gap,
        fig_dir=fig_dir
    )[0]
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
