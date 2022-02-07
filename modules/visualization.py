import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from munging import compute_means, compute_errors


# configure pandas table display
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def sns_styleset():
    """Configure parameters for plotting"""
    sns.set_theme(context='paper',
                  style='whitegrid',
                  palette='deep',
                  font='Arial')
    mpl.rcParams['figure.dpi']        = 300
    mpl.rcParams['axes.linewidth']    = 1
    mpl.rcParams['grid.color']        = '.8'
    mpl.rcParams['axes.edgecolor']    = '.15'
    mpl.rcParams['xtick.bottom']      = True
    mpl.rcParams['ytick.left']        = True
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1
    mpl.rcParams['xtick.color']       = '.15'
    mpl.rcParams['ytick.color']       = '.15'
    mpl.rcParams['xtick.major.size']  = 3
    mpl.rcParams['ytick.major.size']  = 3
    mpl.rcParams['font.size']         = 11
    mpl.rcParams['axes.titlesize']    = 11
    mpl.rcParams['axes.labelsize']    = 10
    mpl.rcParams['legend.fontsize']   = 10
    mpl.rcParams['legend.frameon']    = False
    mpl.rcParams['xtick.labelsize']   = 10
    mpl.rcParams['ytick.labelsize']   = 10


def error_line(df, var, n, xlabel=None, ylabel=None, label=None):
    """Plot the mean of variable 'var' at each match for all account_ids over 'n' matches, with shaded regions indicating SEM"""
    means = compute_means(df, var, n)
    errors = compute_errors(df, var, n)

    xticks = [1]
    xticks.extend(list(range(n // 10,
                             n + n // 10,
                             n // 10 if n // 10 != 0 else 1)))

    plt.plot(np.arange(1, n + 1), means)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.fill_between(np.arange(1, n + 1),
                     [means[i] + errors[i] for i in range(n)],
                     [means[i] - errors[i] for i in range(n)],
                     alpha=0.15)

    
def visualize_auto_elbow(n_clusters, inertias, n_matches, start_gap, end_gap, fig_dir, grad_line, optimal_k):
    """Return elbow plot of k-means cluster inertias"""
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

    plt.ylabel('Inertia')
    plt.xlabel('Cluster N')
    
    fig_name = f'matches_{start_gap}-{n_matches-end_gap}_elbow_plot'
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.png'))
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.tif'), format='tiff')
    plt.show()


def visualize_cluster_overlap(df, group_1, group_2, groups_rmp=None, csv_dir=os.path.join('results', 'tables'), fig_dir=os.path.join('figures', 'appendix', 'tables'), file_name=None):
    """Tabulate relative frequencies of spacing categories group_1 and group_2 against eachother, specify axis labels as dictionary using groups_rmp"""
    group_1_unique = np.sort(df[group_1].unique())
    group_2_unique = np.sort(df[group_2].unique())

    grouped_df = df.groupby([group_1, group_2])['account_id'].agg('nunique')
    X = grouped_df.values.reshape((len(group_1_unique), len(group_2_unique)))
    
    if file_name == None:
        file_name=f'{group_1}_{group_2}_overlap'
    
    # output join frequencies to csv
    grouped_df.to_csv(
       os.path.join(os.getcwd(), csv_dir, f'{file_name}.csv')
    )
    
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
    
    plt.savefig(
        f'{os.path.join(fig_dir, file_name)}.tif',
        bbox_inches='tight'     
    )
    plt.savefig(
        f'{os.path.join(fig_dir, file_name)}.png',
        bbox_inches='tight'
    )
    
    plt.show()
    return None


def curves_visualizer(df, grouper, targets, rows, columns, n_matches, fig_size, fig_dir,
                      fig_name, grouper_rmp=None, target_rmp=None, legend=True, accounts_to_retain=None, xticks=None):
    """Save and show multiplot of trajectories with 95% confidence intervals for each variable in targets"""
    # set up multiplot environment
    fig, axs = plt.subplots(
        rows,
        columns,
        figsize=fig_size,
        sharex=True
    )
    # set plot xticks to range from 1 to 100 unless otherwise specified
    if xticks is None:
        xticks = [1]
        xticks.extend(list(range(10, 110, 10)))
    # iterate over groups in label specified by 'grouper'
    for index_g, group in enumerate(df[grouper].unique()):
        # iterate over each of our target variables
        for index, target in enumerate(targets):
            if accounts_to_retain is not None:
                subsample = df[df['account_id'].isin(accounts_to_retain[target])]
                # means and standard errors at each match for each variable in each of our groups
                means = subsample[subsample[grouper] == group].groupby(
                    'nth_match', sort=False)[target].mean().tolist()
                errors = subsample[subsample[grouper] == group].groupby(
                    'nth_match', sort=False)[target].sem().tolist()
            else:
                means = df[df[grouper] == group].groupby(
                    'nth_match', sort=False)[target].mean().tolist()
                errors = df[df[grouper] == group].groupby(
                    'nth_match', sort=False)[target].sem().tolist()                
            # group labels for legend
            if grouper_rmp is not None:
                label = f'{grouper_rmp[grouper]} {index_g+1}'
            else:
                label = f'{group}'
            # plot trajectories with 95% confidence intervals
            axs[index].plot(means, label=label)
            axs[index].fill_between(
                np.arange(1, n_matches + 1),
                [means[i] + (errors[i]*1.96) for i in range(len(means))],
                [means[i] - (errors[i]*1.96) for i in range(len(means))],
                alpha=0.15,
            )
            # adjust y axis labels for each target variable
            if target_rmp is not None:
                target_label = target_rmp[target]
            else:
                target_label = target
            
            axs[index].set_xticks(xticks)
            axs[index].axvline(95, c='r', linestyle='--')
            axs[index].set_ylabel(target_label)
            axs[index].set_xlabel('Match')
    
    plt.tight_layout()
    if legend:
        plt.legend(
            # loc='center left',
            # bbox_to_anchor=(1, 0.5)
    )
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.png'), bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.tif'), format='tiff', bbox_inches='tight')

    plt.show()
    

def profiles_visualizer(df, grouper, grouper_rmp, dim_reduction, target, fig_size, fig_dir, fig_name, mask=-1,
                        target_rmp=None, cmap='tab10', legend=True, xticks=None):

    """
    """
    # set up multiplot environment
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    axs = axs.flatten()
    cmapper = mpl.cm.get_cmap(cmap)  
    # set plot xticks to range from 1 to 100 unless otherwise specified
    if xticks is None:
        xticks = [1]
        xticks.extend(list(range(10, 110, 10)))

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

            axs[1].plot(nth_matches, means, label=label, color=color)
            axs[1].fill_between(
                [i for i in nth_matches],
                [means[i] + (1.96 * errors[i]) for i in range(len(nth_matches))],
                [means[i] - (1.96 * errors[i]) for i in range(len(nth_matches))],
                alpha=0.25,
            )

            if target_rmp is not None:
                target_label = target_rmp[target]
            else:
                target_label = target
    
    axs[0].set_xlabel('Dimension 1')
    axs[0].set_ylabel('Dimension 2')

    axs[1].set_xticks(xticks)
    axs[1].set_ylabel(target_label)
    axs[1].set_xlabel('Match')
    axs[1].set_xlim(nth_matches.min(), nth_matches.max())
    axs[1].axvline(95, c='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if legend:
        axs[1].legend()
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.png'))
    plt.savefig(os.path.join(fig_dir, f'{fig_name}.tif'), format='tiff')

    plt.show()


class SeabornFig2Grid():
    """Enables seaborn multiplots"""
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
