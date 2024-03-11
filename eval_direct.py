from ast import literal_eval

import seaborn as sns
import torch

import dataset
import eval.dataset
import eval.models as em
import utils
from train_utils import parse_args
from trainer.direct import marginalize_non_existing_filters
from torch.nn.functional import l1_loss


def initialize(args):
    sns.set_style("ticks")
    utils.enforce_reproducibility()
    _, val_dl = dataset.get_data_loaders(
        None,
        f'{args.data_path}/sampled_filters_val.pkl',
        utils.none_literal_eval(args.drop_filters),
        utils.none_literal_eval(args.drop_dust_features),
        args.batch,
        1,
        args.normalize
    )
    model_kwargs = literal_eval(args.model_kwargs)
    model = utils.models_dict[args.model](**model_kwargs)
    model.load_state_dict(
        torch.load(
            f'{args.stats_path}/checkpoint/best_model.pt',
            map_location=torch.device('cpu')
        )
    )
    labels = [
        filter_name for filter_name in dataset.filter_indexes.keys()
        if filter_name not in args.drop_filters
    ]
    model.eval()
    return model, val_dl, labels


def main(args):
    args.batch = 1000
    model, val_dl, labels = initialize(args)
    eval.models.plot_losses(args.stats_path, log_y_scale=False, only_best=True)

    means, l_tris, x_exist = eval.models.get_predictions_direct(model, val_dl)

    mses, mses_dist, nlls, conf_acc, cov_mean, _ = em.get_stats(
        val_dl, means, l_tris, args.min_d, compute_nll=True
    )

    em.plot_mean_stats(mses, args.stats_path, labels, rotate=True)

    em.calibration_plot(conf_acc, val_dl.dataset.x.shape[1], args.stats_path)
    print('NLL', nlls.mean())
    em.err_distribution_plot(
        mses_dist, nlls, args.stats_path, n_bins=500, drop_n_worst=50,
        percentile=0.001
    )
    em.plot_mean_covariance(
        cov_mean, labels, args.stats_path
    )


if __name__ == "__main__":
    data_path = '../data'
    experiment_path = './experiments/direct-likelihood/conv/03'
    args_path = f'@{experiment_path}/args.txt'
    main(parse_args([data_path, experiment_path, args_path]))
