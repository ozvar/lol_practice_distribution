import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def dim_reduction_visualizer(dim_reduction, labels=None, labelling='no_groups',
                             cmap='prism', save_path='results\\figures',
                             figsize=(10, 10), mask=-1, dpi=300, legend=False):
    """
    """
    cmap = matplotlib.cm.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    if labels is not None:
        index_label = 0
        for unique_label in np.unique(labels):

            label_index = np.argwhere(labels == unique_label).flatten()
            if unique_label == mask:
                color = 'whitesmoke'
            else:
                color = cmap(index_label)
                index_label += 1
            ax.scatter(
                dim_reduction[label_index, 0],
                dim_reduction[label_index, 1],
                s=0.05,
                marker='o',
                color=color,
                label=f'{labelling} {unique_label}',
            )
    else:
        ax.scatter(
            dim_reduction[:, 0],
            dim_reduction[:, 1],
            s=0.05,
            marker='o',
            color='steelblue',
            label='Global'
        )
    if legend:
        ax.legend(
            markerscale=10,
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.savefig(
        f'{save_path}\\{labelling}.svg',
        dpi=dpi,
        bbox_inches='tight'
    )
    plt.show()
    return None


def curves_visualizer(df, grouper, targets, rows, columns, figsize=(10, 5),
                      save_path='results\\figures', dpi=300, grouper_rmp=None,
                      target_rmp=None, cmap='prism', legend=True,
                      accounts_to_retain=None):
    """
    """
    fig, axs = plt.subplots(
        rows,
        columns,
        figsize=figsize,
    )
    axs = axs.flatten()
    cmapper = matplotlib.cm.get_cmap(cmap)

    for index_g, group in enumerate(df[grouper].unique()):

        for index, target in enumerate(targets):
            
            if accounts_to_retain is not None:
                filtered_df = df[
                    df['account_id'].isin(accounts_to_retain[target])
                ]
            else:
                filtered_df = df
            
            nth_matches = np.sort(filtered_df['nth_match'].unique())
            means = filtered_df[filtered_df[grouper] == group].groupby(
                'nth_match')[target].mean().values
            errors = filtered_df[filtered_df[grouper] == group].groupby(
                'nth_match')[target].sem().values

            if grouper_rmp is not None:
                label = f'{grouper_rmp[grouper]} {group}'
            else:
                label = f'{grouper} {group}'

            axs[index].plot(
                nth_matches,
                means, 
                label=label, 
                c=cmapper(index_g)
            )
            axs[index].fill_between(
                [i for i in nth_matches],
                [means[i] + (1.96 * errors[i]) for i in range(len(nth_matches))],
                [means[i] - (1.96 * errors[i]) for i in range(len(nth_matches))],
                alpha=0.25,
                color=cmapper(index_g)
            )

            if target_rmp is not None:
                target_label = target_rmp[target]
            else:
                target_label = target

            axs[index].set_ylabel(target_label)
            axs[index].set_xlabel('Match')
            axs[index].set_xlim(nth_matches.min(), nth_matches.max())
            axs[index].axvline(95, c='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if legend:
        plt.legend(
            # loc='center left',
            # bbox_to_anchor=(1, 0.5)
    )
    plt.savefig(
        f'{save_path}\\{grouper}_curves.svg',
        dpi=dpi,
        bbox_inches='tight'
    )

    plt.show()
    return None


def profiles_visualizer(df, target, dim_reduction, grouper='global',
                        figsize=(10, 5), mask=-1, save_path='results\\figures',
                        dpi=300, grouper_rmp={'global': 'Global'},
                        target_rmp=None, cmap='prism', legend=True):
    """
    """
    fig, axs = plt.subplots(
        1,
        2,
        figsize=figsize,
    )
    axs = axs.flatten()
    cmapper = matplotlib.cm.get_cmap(cmap)
    
    nth_matches = np.sort(df['nth_match'].unique())
    labels = df.groupby('account_id')[grouper].agg(['unique']).values.tolist()
    labels = np.array(labels).flatten()
    index_color_label = 0
    for unique_label in np.unique(labels):

        label_index = np.argwhere(labels == unique_label).flatten()
        if unique_label == mask:
            color = 'whitesmoke'
        else:
            color = cmapper(index_color_label)
            index_color_label += 1
        axs[0].scatter(
            dim_reduction[label_index, 0],
            dim_reduction[label_index, 1],
            s=0.025,
            marker='o',
            color=color,
            label=f'{grouper_rmp[grouper]} {unique_label+1}',
        )

        if unique_label != mask:
            means = df[df[grouper] == unique_label].groupby(
                'nth_match')[target].mean().values
            errors = df[df[grouper] == unique_label].groupby(
                'nth_match')[target].sem().values

            label = f'{grouper_rmp[grouper]} {unique_label+1}'

            axs[1].plot(
                nth_matches,
                means, 
                label=label, 
                c=color
            )
            axs[1].fill_between(
                [i for i in nth_matches],
                [means[i] + (1.96 * errors[i]) for i in range(len(nth_matches))],
                [means[i] - (1.96 * errors[i]) for i in range(len(nth_matches))],
                alpha=0.25,
                color=color
            )

            if target_rmp is not None:
                target_label = target_rmp[target]
            else:
                target_label = target

    axs[0].set_xlabel('Dimension 1')
    axs[0].set_ylabel('Dimension 2')

    axs[1].set_ylabel(target_label)
    axs[1].set_xlabel('Match')
    axs[1].set_xlim(nth_matches.min(), nth_matches.max())
    axs[1].axvline(95, c='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if legend:
        axs[1].legend(
            # loc='center left',
            # bbox_to_anchor=(1, 0.5)
        )
    plt.savefig(
        f'{save_path}\\{grouper}_profile.svg',
        dpi=dpi,
        bbox_inches='tight'
    )

    plt.show()
    return None


def visualize_auto_elbow(n_clusters, inertias, grad_line, optimal_k,
                         save_path, dpi=300):
    """
    """
    optimal_index = np.where(np.array(n_clusters) == optimal_k)
    optimal_index = optimal_index[0][0]

    plt.figure(figsize=(10, 10))
    # draw the traditional elbow plot
    plt.plot(
        n_clusters,
        inertias,
        marker='*',
        color='b'
    )
    # draw the overall gradient of the inertia with respect to the
    # number of clusters
    plt.plot(
        [n_clusters[0], n_clusters[-1]],
        [inertias[0], inertias[-1]],
        color='r',
        linestyle='--',
        alpha=0.5
    )
    # highlight the point of maximum curvature
    plt.vlines(
        optimal_k,
        inertias[optimal_index],
        grad_line[optimal_index],
        color='r',
        linestyle='--'
    )

    plt.title(f'Optimal K={optimal_k}')
    plt.ylabel('Inertia')
    plt.xlabel('Number of Centroids')

    plt.savefig(
        f'{save_path}.svg',
        dpi=dpi,
        bbox_inches='tight'
    )


def visualize_cluster_overlap(df, group_1, group_2, groups_rmp=None):
    """
    """
    group_1_unique = np.sort(df[group_1].unique())
    group_2_unique = np.sort(df[group_2].unique())

    grouped_df = df.groupby([group_1, group_2])['account_id'].agg('nunique')
    X = grouped_df.values.reshape((len(group_1_unique), len(group_2_unique)))

    sns.heatmap(
        np.round(X / X.sum(), 2),
        cmap='RdBu_r',
        annot=True
    )

    if groups_rmp is not None:
        x_label = groups_rmp[group_2]
        y_label = groups_rmp[group_1]
    else:
        x_label = group_2
        y_label = group_1

    plt.ylabel(y_label)
    plt.yticks(
        [tick + 0.5 for tick in range(len(group_1_unique))],
        group_1_unique
    )

    plt.xlabel(x_label)
    plt.xticks(
        [tick + 0.5 for tick in range(len(group_2_unique))],
        group_2_unique
    )

    grouped_df.to_csv(
       f'results\\tables\\{group_1}_{group_2}_overlap.csv'
    )
    plt.savefig(
        f'results\\figures\\{group_1}_{group_2}_overlap.svg',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    return None


def bayesian_comparison(group_1, group_2, samples):
    """
    """
    dist_differences = samples[:, group_1] - samples[:, group_2]

    plt.figure(figsize=(5, 5))

    sns.kdeplot(dist_differences)

    plt.title(f'GPM variation cluster {group_1} minus cluster {group_2}')
    plt.xlabel('Difference')
    plt.ylabel('Density')
