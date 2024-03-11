import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dataset
import eval.models


def plot_worst_x(worst_x, x, labels, save_path):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7)) # 6.4
    metadata = [
        (x, 'tab:blue', 'tab:orange', 'Whole dataset'),
        (worst_x, 'tab:red', 'tab:green', 'Worst samples')
    ]
    print(labels, worst_x.shape)
    for i in range(2):
        n_features = metadata[i][0].shape[1]
        n_samples = metadata[i][0].shape[0]
        print(torch.arange(n_features).repeat(n_samples).shape)
        print(metadata[i][0].flatten().shape)
        if i == -1:
            sns.histplot(
                x=torch.arange(n_features).repeat(n_samples),
                y=metadata[i][0].flatten(),
                # fill=True,
                bins='auto',
                ax=ax[i],
                cbar=True,
                # color=metadata[i][0],
                # label='histogram'
                # levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                # thresh=0.005,
                # label='samples distribution'
            )
        else:
            sns.stripplot(
                x=torch.arange(n_features).repeat(n_samples),
                y=metadata[i][0].flatten(),
                # fill=True,
                # bins='auto',
                ax=ax[i],
                orient='v',
                linewidth=1,
                # cbar=True,
                color=metadata[i][1],
                size=3
                # label='histogram'
                # levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                # thresh=0.005,
                # label='samples distribution'
            )
        """
        mean = metadata[i][0].mean(axis=0)
        std = metadata[i][0].std(axis=0)
        ax[i].vlines(
            np.arange(x.shape[1]), mean - std, mean + std,
            color=metadata[i][2], label='$\pm$ std'
        )
        ax[i].plot(
            mean, marker='o', label='mean', linestyle='None',
            color=metadata[i][1]
        )
        """
        ax[i].set_xticks(np.arange(x.shape[1]))
        ax[i].set_xticklabels(labels, rotation=90)
        # ax[i].legend()
        ax[i].set_title(metadata[i][3])
        ax[i].set_ylabel('AB magnitude')
        ax[i].invert_yaxis()
    fig.tight_layout()
    fig.savefig(f'{save_path}/plots/worst-samples/worst-x.png', dpi=300)


def plot_worst_y(
        samples, y_all, save_path, kind='hist', color='tab:blue', f_name=''
):
    y_feature_names_full = [
        'dust mass', 'grain size', 'silicate fraction', "dust temperature",
        'clump factor', "supernova luminosity", "supernova temperature"
    ]

    fig, axs = plt.subplots(3, 7, figsize=(40, 13)) # 16 30
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    for i, ax in enumerate(axs.flat):
        a, b = pairs[i]
        print(i)
        if kind == 'hist':
            sns.histplot(
                x=samples[:, a],
                y=samples[:, b],
                # fill=True,
                bins='auto',
                ax=ax,
                cbar=True,
                color=color,
                # label='histogram'
                # levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                # thresh=0.005,
                # label='samples distribution'
            )
        elif kind == 'kde':
            sns.kdeplot(
                x=samples[:, a],
                y=samples[:, b],
                fill=True,
                ax=ax,
                cbar=True,
                color=color,
                # label='KDE'
                # levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                # thresh=0.005,
                # label='samples distribution'
            )
        ax.scatter(
            samples[:, a], samples[:, b], color='tab:orange',
            label='exact value', marker='d', edgecolor='black',
        )
        ax.set_xlabel(y_feature_names_full[a])
        ax.set_ylabel(y_feature_names_full[b])
        ax.set_xlim(y_all[:, a].min(), y_all[:, a].max())
        ax.set_ylim(y_all[:, b].min(), y_all[:, b].max())
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        f'{save_path}/plots/worst-samples/{f_name}-samples.png',
        dpi=200
    )


def plot_worst_z(worst_z, save_path):
    fig, ax = plt.subplots()
    sns.histplot(worst_z[:, 0], ax=ax)
    ax.set_xlabel('$z$')
    fig.tight_layout()
    fig.savefig(f'{save_path}/plots/worst-samples/z-stats.png', dpi=300)


def get_z_fractions(worst_y, y):
    worst_y, reverse_idxs, counts = torch.unique(
        worst_y, dim=0, return_inverse=True, return_counts=True
    )
    y, _, y_counts = torch.unique(
        y, dim=0, return_inverse=True, return_counts=True
    )
    res = []
    for i in range(worst_y.shape[0]):
        for j in range(y.shape[0]):
            if (worst_y[i, :] == y[j, :]).all():
                res.append(counts[i] / y_counts[j])
    return torch.tensor(res)


def save_example(x, mean, cov, mahalanobis, nll, idx, labels, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.5))  # likelihood model
    ax[0].plot(
        x, marker='o', label='true value',
        color='tab:orange'
    )
    ax[0].plot(
        mean, marker='o', label='prediction',
        color='tab:blue'
    )
    mae = torch.abs(x - mean).mean()
    ax[0].set_title(f'Mahalanobis distance: {mahalanobis:.2f}, NLL: {nll:.2f}, MAE: {mae:.2f}')
    ax[0].legend()
    ax[0].set_xticks(np.arange(x.shape[0]))
    ax[0].set_xticklabels(labels, rotation=90)
    ax[0].invert_yaxis()
    #fig.tight_layout()
    # fig.savefig(f'{save_path}/plots/worst-samples/tmp-duplicates/mean-{idx}.png', dpi=300)

    # fig, ax = plt.subplots(figsize=(7, 5.5))  # likelihood model
    # fig, ax = plt.subplots(figsize=(4, 3))  # inference model
    sns.heatmap(
        cov,
        ax=ax[1],
        xticklabels=labels,
        yticklabels=labels,
        # vmin=0.
    )
    ax[1].set_aspect('equal', 'box')
    fig.tight_layout()
    fig.savefig(
        f'{save_path}/plots/worst-samples/tmp/{idx}.png',
        dpi=300
    )

def plot_worst_samples(
        nlls, x, y, z, mean, cov, labels, save_path, percentile=0.01,
        n_examples=50
):
    filter_idxs = torch.tensor([
        dataset.filter_indexes[filter_name] for filter_name in labels
    ])
    mean = mean * dataset.train_data_stats['x_stats'][1][filter_idxs] + \
        dataset.train_data_stats['x_stats'][0][filter_idxs]
    nlls_argsorted = torch.argsort(nlls, descending=False)
    n_samples = -int(percentile * nlls.shape[0])
    worst_z = z[nlls_argsorted, :][n_samples:]
    worst_z_argsorted = torch.argsort(worst_z, dim=0).squeeze(1)
    worst_y_ = y[nlls_argsorted, :][n_samples:][worst_z_argsorted]
    worst_y, reverse_idxs, counts = torch.unique(
        worst_y_, dim=0, return_inverse=True, return_counts=True
    )
    # print(torch.cat([get_z_fractions(worst_y_, y).unsqueeze(1), counts.unsqueeze(1)], dim=1))
    print('...')
    idxs = []
    # print(counts)
    # print(z.shape, torch.unique(z, dim=0).shape)
    for i in range(worst_y.shape[0]):
        idxs.append(torch.nonzero(reverse_idxs == i)[0][0])
    idxs = torch.tensor(idxs)  # torch.arange(worst_z.shape[0])
    worst_x = x[nlls_argsorted, :][n_samples:][idxs]
    print(worst_x.shape, worst_y.shape, worst_z.shape)
    plot_worst_x(worst_x, x, labels, save_path)
    plot_worst_y(worst_y, y, save_path, kind='kde', f_name='worst-y')
    plot_worst_z(worst_z, save_path)

    nll = nlls[nlls_argsorted][n_samples:][idxs]
    nll_argsorted = torch.argsort(nll)
    nll = nll[nll_argsorted]
    mean = mean[nlls_argsorted, :][n_samples:][idxs][nll_argsorted]
    cov = cov[nlls_argsorted, :, :][n_samples:][idxs][nll_argsorted]
    worst_x = worst_x[nll_argsorted]
    print(nll)
    for i in range(n_examples):
        sample_mean = mean[-i-1].unsqueeze(0)
        sample_cov = cov[-i-1].unsqueeze(0)
        sample_x = worst_x[-i-1].unsqueeze(0)
        mahalanobis = eval.models.get_mahalanobis_distance(
            sample_mean, sample_cov, sample_x
        )
        save_example(
            sample_x[0], sample_mean[0], sample_cov[0], mahalanobis[0],
            nll[-i-1], i, labels, save_path
        )


    """
    mean = samples_x.mean(axis=0)
    std = samples_x.std(axis=0)
    plt.plot(mean, marker='o', label='mean', linestyle='None')
    plt.vlines(np.arange(mean.shape[0]), mean - std, mean + std)
    plt.show()
    fig, ax = plt.subplots()
    sns.histplot(samples_z[:, 0], ax=ax)
    ax.set_xlabel('$z$')
    fig.tight_layout()
    fig.savefig(f'{args.stats_path}/plots/worst-samples/z-stats.png', dpi=300)
    plt.show()
    print('samples z', samples_z.mean(), samples_z.std())
    print(samples.shape)
    print(true_y.min(dim=0), true_y.max(dim=0))
    plot_samples(args, samples, true_y, kind='kde', f_name='worst-y')
    plot_x_samples(true_data, samples_x, args.stats_path)
    """
