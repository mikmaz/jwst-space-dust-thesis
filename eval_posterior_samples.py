import pickle
import time

import numpy as np

from cvae_utils import parse_args as parse_args_cvae
from eval_cvae import initialize as initialize_cvae
from eval_direct import initialize
from posterior_samples import check_mixing
from train_utils import parse_args
import seaborn as sns
import matplotlib.pyplot as plt
import dataset


def plot_dist(
        samples, data, save_path, true_y=None, color='tab:blue', kind='hist',
        suptitle=None
):
    fig, axs = plt.subplots(7, 3, figsize=(16, 30))
    pairs = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    for i, ax in enumerate(axs.flat):
        a, b = pairs[i]
        """
        ax.scatter(
            y[kl_divs_arg_sorted[:k], a],
            y[kl_divs_arg_sorted[:k], b],
            cmap='viridis',
        )
        """
        if kind == 'hist':
            sns.histplot(
                x=samples[:, a],
                y=samples[:, b],
                # fill=True,
                bins='auto',
                ax=ax,
                cbar=True,
                color=color,
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
                # levels=[0.2, 0.4, 0.6, 0.8, 1.0],
                # thresh=0.005,
                # label='samples distribution'
            )
        if true_y is not None:
            ax.scatter(
                true_y[:, a], true_y[:, b], color='tab:orange',
                label='true data', marker='d', edgecolor='black'
            )
            ax.legend()
        ax.set_xlabel(dataset.y_feature_names_full[a])
        ax.set_ylabel(dataset.y_feature_names_full[b])
        ax.set_xlim(data.y[:, a].min(), data.y[:, a].max())
        ax.set_ylim(data.y[:, b].min(), data.y[:, b].max())
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=30)
        fig.tight_layout(rect=[0, 0.03, 1., 0.95])
    else:
        fig.tight_layout()
    # fig.suptitle(f'center={center}, k={k}')
    fig.savefig(
        save_path,
        dpi=200
    )


def plot_samples(hmc_samples, data, save_path, indexes):
    for hmc_sample in [hmc_samples[i] for i in indexes]:
        sample = hmc_sample['hmc_runs'][-1].reshape(-1, data.y.shape[1])
        sample_save_path = save_path + f'/{hmc_sample["idx"]}.png'
        print('Plotting sample', hmc_sample['idx'])
        plot_dist(
            sample, data, sample_save_path,
            true_y=data.y[hmc_sample["idx"]].unsqueeze(0), kind='hist',
            suptitle=f"KDE of sample {hmc_sample['idx']}"
        )


def filter_samples(hmc_samples):
    proper_samples = [
        sample for sample in hmc_samples if check_mixing(sample['hmc_runs'][-1])
    ]
    return proper_samples


def print_avg_runtime(samples):
    runtimes_2k = np.array([sample['runtimes'][0] for sample in samples])
    runtimes_4k = np.array([
        sample['runtimes'][1] for sample in samples
        if len(sample['runtimes']) > 1
    ])
    print(len(samples[0]['hmc_runs'][0][0]))
    runtimes_4k = np.array([0.]) if len(runtimes_4k) == 0 else runtimes_4k
    runtimes_all = np.array([sum(sample['runtimes']) for sample in samples])
    avg_2k_runtime = time.strftime(
        '%H:%M:%S', time.gmtime(np.mean(runtimes_2k))
    )
    avg_4k_runtime = time.strftime(
        '%H:%M:%S', time.gmtime(np.mean(runtimes_4k))
    )
    total = np.sum(runtimes_2k) + np.sum(runtimes_4k)
    print('Number of samples:', len(samples))
    print(f'Average runtime for 2k samples:', avg_2k_runtime)
    print(f'Average runtime for 4k samples:', avg_4k_runtime)
    print(f'Total runtime:', total / 3600, 'hours')
    print(f'Mean runtime:', np.mean(runtimes_all / 60))
    print(f'SD runtime:', np.std(runtimes_all / 60))


def main(stats_path, args, cvae, marginalize=False):
    model, val_dl, labels = initialize(
        args
    ) if not cvae else initialize_cvae(args)

    with open(f'{stats_path}/results_combined.pkl', 'rb') as f:
        hmc_samples = pickle.load(f)
    print_avg_runtime(hmc_samples)

    hmc_samples = filter_samples(hmc_samples)
    print_avg_runtime(hmc_samples)
    plot_samples(
        hmc_samples, val_dl.dataset, f'{stats_path}/plots',
        [i for i in range(len(hmc_samples))]
    )


if __name__ == "__main__":
    data_path = '../data/full'
    experiment_path = 'experiments/hmc/direct-linear/submission-02'  # sys.argv[2]
    model_path = 'experiments/direct-likelihood/linear/08'  # sys.argv[3]
    cvae_ = False
    marg = False

    sns.set_style("ticks")
    args_path = f'@{model_path}/args.txt'
    if cvae_:
        main(
            experiment_path,
            parse_args_cvae([data_path, model_path, args_path]), cvae_,
            marg
        )
    else:
        main(
            experiment_path, parse_args([data_path, model_path, args_path]),
            cvae_, marg
        )
