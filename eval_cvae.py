from cvae_utils import make_net, parse_args
import utils
import dataset
from models.cvae import CVAE
import eval.models as em
import torch
from trainer.cvae import CVAETrainer
import seaborn as sns

def initialize(args):
    sns.set_style("ticks")
    utils.enforce_reproducibility()
    _, val_dl = dataset.get_data_loaders(
        None,
        f'{args.data_path}/sampled_filters_val.pkl',
        utils.none_literal_eval(args.drop_filters),
        utils.none_literal_eval(args.drop_dust_features),
        args.batch,
        args.n_workers,
        args.normalize
    )
    direct_net = make_net(args.direct_net, args.direct_kwargs)
    prior_net = make_net(args.prior_net, args.prior_kwargs)
    rec_net = make_net(args.recognition_net, args.recognition_kwargs)
    gen_net = make_net(args.gen_net, args.gen_kwargs)
    model = CVAE(
        direct_net, prior_net, rec_net, gen_net, args.recurrent_connection
    )
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
    args.batch = 100
    args.n_latent_samples = 50
    args.n_workers = 1
    model, val_dl, labels = initialize(args)
    em.plot_losses(args.stats_path, log_y_scale=False, only_best=True)
    # em.plot_is_est_vars(model, val_dl, args.min_d, (100, 1000, 100))
    nlls, var_est = em.get_nll_is_est(
        model, val_dl, args.n_latent_samples, args.min_d
    )
    print('IS NLL', nlls.mean())
    means, covs, prior_params, rec_params, _ = em.get_predictions_cvae(
        model, val_dl, args.n_latent_samples, args.min_d
    )

    mses, mses_dist, _, conf_acc, cov_mean, _ = em.get_stats(
        val_dl, means, None, args.min_d, compute_nll=False, covs=covs
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


    # results = torch.load(f'{args.stats_path}/stats/eval_stats.pt')
    # losses = results['loss']
    # print(losses.shape, losses.mean(dim=0))
    # torch.save(results, f'{args.stats_path}/stats/eval_stats.pt')

    """
    loss_kwargs = {
        'min_d': args.min_d, 'l_tri_loss': args.l_tri_loss,
        'gsnn_frac': args.gsnn_frac, 'n_samples': args.n_latent_samples
    }
    trainer = CVAETrainer(
        None, val_dl, model, torch.device('cpu'), args.lr, loss_kwargs
    )
    print(trainer.evaluate()[0].mean())
    """


if __name__ == "__main__":
    data_path = '../data'
    experiment_path = 'experiments/cvae/linear/new/01'
    args_path = f'@{experiment_path}/args.txt'
    args = parse_args([data_path, experiment_path, args_path])
    main(args)
