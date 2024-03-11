import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
from torcheval.metrics import R2Score
from scipy.stats import chi2
import seaborn as sns
from trainer.direct import marginalize_non_existing_filters
from torch.nn.functional import mse_loss


def bmv(m, v):
    return torch.bmm(m, v.unsqueeze(2)).squeeze(2)


def b_dot(v_1, v_2):
    return (v_1 * v_2).sum(dim=1)


def plot_loss(
        path, losses, min_val, colors=('tab:blue', 'tab:orange'),
        labels=('training', 'validation'), xlabel='Epoch', ylabel='Loss',
        f_name='losses.png', log_y_scale=False
):
    fig, ax = plt.subplots()
    for i in range(2):
        ax.plot(losses[:, i], color=colors[i], label=labels[i])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y_scale:
        ax.set_yscale('log')
    ax.legend()
    ax.set_title(f'Min validation value: {min_val:.3f}')
    fig.tight_layout()
    fig.savefig(f'{path}/plots/{f_name}', dpi=600)


def plot_losses(path, only_best=False, log_y_scale=False):
    losses = np.loadtxt(f'{path}/losses.txt', delimiter=',', skiprows=1)
    nll_losses = losses[:, ::2]
    mae_losses = losses[:, 1::2]
    min_loss_idx = np.argmin(nll_losses[:, 1])
    if only_best:
        print(f'Best validation loss: {nll_losses[min_loss_idx, 1]}')
        nll_losses = nll_losses[:min_loss_idx + 1, :]
        mae_losses = mae_losses[:min_loss_idx + 1, :]
    plot_loss(
        path, nll_losses, nll_losses[min_loss_idx, 1],
        f_name='training-loss.png', log_y_scale=log_y_scale
    )
    plot_loss(
        path, mae_losses, mae_losses[min_loss_idx, 1],
        f_name='training-mae.png', colors=('tab:red', 'tab:green'),
        ylabel='MAE', log_y_scale=log_y_scale
    )


def plot_mean_stats(mae, save_path, labels, rotate=False):
    n_features = mae.shape[0]
    mean_x_ticks = np.arange(-0.5, n_features - 0.3, 0.1)
    # metric = R2Score(multioutput="raw_values")
    # metric.update(mean, true_data)
    # r2_scores = metric.compute()
    plots_metadata = [
        (mae, 'MAE', 'tab:red'), (mae, r'$R^2$', 'tab:orange')
    ]
    for i in range(2):
        fig, ax = plt.subplots(figsize=(6.4, 3.5))  # likelihood models
        # fig, ax = plt.subplots(figsize=(4, 2.5))  # inference models
        stat = plots_metadata[i][0]
        ax.bar(
            np.arange(n_features), stat,
            label=f'{plots_metadata[i][1]} per feature', tick_label=labels,
            color=plots_metadata[i][2], alpha=0.8
        )
        ax.plot(
            mean_x_ticks, torch.ones([mean_x_ticks.shape[0]]) * stat.mean(),
            label=f'mean {plots_metadata[i][1]}', color='black', linewidth=3,
            linestyle=(0, (5, 1))
        )
        # print(stat.max() * 1.1)
        # ax.set_ylim(top=0.1460)
        ax.set_ylabel(plots_metadata[i][1])
        # ax1[0].set_title('Mean absolute error per feature (and mean)')
        if rotate:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/plots/{plots_metadata[i][1]}-mean-stats.png',
            dpi=600
        )


def get_mahalanobis_distance(mean, cov, true_data):
    diff = true_data - mean
    cov_inv = torch.inverse(cov)
    return torch.sqrt(b_dot(diff, bmv(cov_inv, diff)))


def get_confidence_accuracies(mean, cov, true_data):
    mahalanobis_dists = get_mahalanobis_distance(mean, cov, true_data)
    df = mean.shape[1]
    thresholds = [i * 0.1 for i in range(1, 10)]
    accuracies = []
    for conf in thresholds:
        accuracies.append(
            (mahalanobis_dists ** 2 <= chi2.ppf(conf, df)).float().unsqueeze(1)
        )
    return torch.cat(accuracies, dim=1)


def calibration_plot(accuracies, df, save_path):
    thresholds = [i * 0.1 for i in range(1, 10)]
    accuracies = accuracies.mean(dim=0)

    err_bottom = []
    err_height = []
    for i in range(len(thresholds)):
        err = accuracies[i] - thresholds[i]
        err_bottom.append(thresholds[i] if err > 0 else accuracies[i])
        err_height.append(abs(err))
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(
        thresholds, accuracies, width=0.1, edgecolor='black', color='tab:blue',
        label='measurements'
    )
    ax.bar(
        thresholds, err_height, bottom=err_bottom, width=0.1, color='tab:red',
        alpha=0.4, hatch='/', edgecolor='tab:red', label='calibration gap'
    )
    ax.plot(
        thresholds, thresholds, c='black', linestyle='--',
        label='perfect calibration'
    )
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_aspect('equal', 'box')
    # ax.set_title('Reliability diagram')
    fig.tight_layout()
    fig.savefig(f'{save_path}/plots/reliability-diagram.png', dpi=600)


def err_distribution_plot(
        mae, nll, save_path, drop_n_worst=0, percentile=None, n_bins=500):
    plot_metadata = [(nll, 'tab:purple', 'NLL'), (mae, 'tab:green', 'MAE')]
    for i in range(2):
        stat = plot_metadata[i][0]
        stat = torch.sort(stat)[0]
        if drop_n_worst > 0:
            stat = stat[:-drop_n_worst]
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        if percentile is not None:
            ax.axvspan(
                stat[-int(percentile * stat.shape[0])], stat[-1], color='gray',
                alpha=0.2, hatch='/',
                label=f'bottom {percentile * 100}%', zorder=-1
            )
            ax.legend()
        sns.histplot(
            stat, log_scale=(False, True), ax=ax, color=plot_metadata[i][1],
            zorder=15, bins=n_bins
        )
        ax.set_xlabel(plot_metadata[i][2])
        ax.set_title(f'Average {plot_metadata[i][2]}: {stat.mean():.4f}')
        fig.tight_layout()
        fig.savefig(
            f'{save_path}/plots/{plot_metadata[i][2]}-err-dists.png',
            dpi=600
        )


def plot_mean_covariance(mean_cov, labels, save_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))  # likelihood model
    # fig, ax = plt.subplots(figsize=(4, 3))  # inference model
    print(mean_cov)
    sns.heatmap(
        mean_cov,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
        # vmin=0.
    )
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    fig.savefig(
        f'{save_path}/plots/mean-cov.png',
        dpi=600
    )


def log_subtract(a, b):
    log_min = torch.minimum(a, b)
    log_max = torch.maximum(a, b)
    # print(torch.log(1 - torch.exp(log_min - log_max)))
    return log_max + torch.log(1 - torch.exp(log_min - log_max))


def is_est(model, x, y, z, mask, n_samples, min_d):
    mean, l_tri_flat, prior_param, rec_param, samples = model(
        x, y, z, n_samples=n_samples, sample_from_prior=False,
        ret_latent_samples=True
    )
    x_marg = x[:, mask]
    mean_marg = mean[:, :, mask]

    log_p = -get_cvae_loss(
        x_marg, mean_marg, l_tri_flat, prior_param, rec_param, None,
        n_samples, min_d, mask=mask, n_features=x.shape[1], dont_reduce=True
    )[0].transpose(0, 1)
    prior_dist = torch.distributions.Normal(
        prior_param[0], prior_param[1]
    )
    prior_log_p = prior_dist.log_prob(samples).sum(-1)

    rec_dist = torch.distributions.Normal(
        rec_param[0], rec_param[1]
    )
    rec_log_p = rec_dist.log_prob(samples).sum(-1)

    if len(prior_log_p.shape) == 1:
        prior_log_p = prior_log_p.unsqueeze(1)
        rec_log_p = rec_log_p.unsqueeze(1)

    # print(log_p.shape, prior_log_p.shape, rec_log_p.shape)
    # print((log_p + prior_log_p - rec_log_p).shape)
    nll = -torch.logsumexp(
        log_p + prior_log_p - rec_log_p, dim=0
    ) + np.log(n_samples)

    return nll


def get_nll_is_est(model, val_dl, n_samples, min_d, n_batches=None):
    dataset_size = val_dl.dataset.x.shape[0]
    out_features = val_dl.dataset.x.shape[1]
    is_var_ests = torch.zeros(dataset_size)
    nlls = torch.zeros(dataset_size)
    i = 0
    model.eval()
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for x, y, z, x_exist in pbar:
                mean, l_tri_flat, prior_param, rec_param, samples = model(
                    x, y, z, n_samples=n_samples, sample_from_prior=True,
                    ret_latent_samples=True
                )
                masks = marginalize_non_existing_filters(x_exist)
                c = 0
                batch_size = x.shape[0]
                if n_batches is not None and i / batch_size >= n_batches:
                    break
                samples = samples.reshape(n_samples, batch_size, -1)
                for m in range(len(masks)):
                    marg_mask = masks[m][0]
                    idx_mask = masks[m][1]
                    x_marg = x[idx_mask][:, marg_mask]
                    mean_marg = mean[:, idx_mask][:, :, marg_mask]
                    l_flat_filtered = l_tri_flat[:, idx_mask]
                    samples_filtered = samples[:, idx_mask]

                    prior_params_filtered = (
                        prior_param[0][idx_mask], prior_param[1][idx_mask]
                    )
                    rec_params_filtered = (
                        rec_param[0][idx_mask], rec_param[1][idx_mask]
                    )
                    log_p = -get_cvae_loss(
                        x_marg, mean_marg, l_flat_filtered,
                        prior_params_filtered, rec_params_filtered, None,
                        n_samples, min_d, mask=marg_mask,
                        n_features=out_features, dont_reduce=True
                    )[0].transpose(0, 1)
                    prior_dist = torch.distributions.Normal(
                        prior_params_filtered[0], prior_params_filtered[1]
                    )
                    prior_log_p = prior_dist.log_prob(samples_filtered).sum(-1)
                    rec_dist = torch.distributions.Normal(
                        rec_params_filtered[0], rec_params_filtered[1]
                    )
                    rec_log_p = rec_dist.log_prob(samples_filtered).sum(-1)

                    posts = log_p  # + prior_log_p - rec_log_p
                    nll = -torch.logsumexp(posts, dim=0) + np.log(n_samples)

                    log_posteriors = -nll.repeat(n_samples, 1)
                    # print(posts[:, 0], log_posteriors[:, 0])
                    var_est = torch.logsumexp(
                        2 * log_subtract(posts, log_posteriors), dim=0
                    ) - np.log(n_samples)
                    # var_est = torch.var(-posts, dim=0)
                    var_est = var_est / 2 - np.log(np.sqrt(n_samples))
                    marg_idx_n = x_marg.shape[0]
                    nlls[i + c:i + c + marg_idx_n] = nll
                    is_var_ests[i + c:i + c + marg_idx_n] = var_est
                    c += marg_idx_n
                i += batch_size

    return nlls, var_est


def plot_is_est_vars(model, val_dl, min_d, n_samples, n_batches=10):
    # start, stop, step = n_samples
    var_ests = []
    for i in range(*n_samples):
        _, var_est = get_nll_is_est(
            model, val_dl, i, min_d, n_batches=n_batches
        )
        # print(var_est[:n_batches * val_dl.batch_size])
        var_ests.append(var_est[:n_batches * val_dl.batch_size].mean().item())
    print(var_ests)
    fig, ax = plt.subplots()
    ax.plot([i for i in range(*n_samples)], var_ests)
    plt.show()


def get_predictions_cvae(model, val_dl, n_samples, min_d):
    dataset_size = val_dl.dataset.x.shape[0]
    out_features = val_dl.dataset.x.shape[1]

    means = torch.zeros((dataset_size, out_features))
    covs = torch.zeros(
        (dataset_size, out_features, out_features)
    )
    prior_params = torch.zeros((2, dataset_size, model.latent_size))
    rec_params = torch.zeros((2, dataset_size, model.latent_size))
    nlls = torch.zeros(dataset_size)
    i = 0
    model.eval()
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for x, y, z, x_exist in pbar:
                mean, l_tri_flat, prior_param, rec_param = model(
                    x, y, z, n_samples=n_samples, sample_from_prior=True,
                )
                masks = marginalize_non_existing_filters(x_exist)
                c = 0
                for m in range(len(masks)):
                    marg_mask = masks[m][0]
                    idx_mask = masks[m][1]
                    x_marg = x[idx_mask][:, marg_mask]
                    mean_marg = mean[:, idx_mask][:, :, marg_mask]
                    l_flat_filtered = l_tri_flat[:, idx_mask]
                    prior_params_filtered = (
                        prior_param[0][idx_mask], prior_param[1][idx_mask]
                    )
                    rec_params_filtered = (
                        rec_param[0][idx_mask], rec_param[1][idx_mask]
                    )
                    nll = get_cvae_loss(
                        x_marg, mean_marg, l_flat_filtered,
                        prior_params_filtered, rec_params_filtered, None,
                        n_samples, min_d, mask=marg_mask,
                        n_features=out_features
                    )[0][:, 0]
                    marg_idx_n = x_marg.shape[0]
                    nlls[i + c:i + c + marg_idx_n] = nll
                    c += marg_idx_n

                batch_size = x.shape[0]
                l_tri_flat = l_tri_flat.transpose(0, 1).reshape(
                    batch_size * n_samples, -1
                )
                cov = reconstruct_cov(l_tri_flat, out_features, min_d)[0]
                cov = cov.reshape(
                    batch_size, n_samples, out_features, out_features
                ).transpose(0, 1)
                var_mean = torch.diag_embed(mean.var(dim=0))
                cov = cov.mean(dim=0) + var_mean

                means[i:i + batch_size] = mean.mean(dim=0)
                covs[i:i + batch_size] = cov
                prior_params[0, i:i + batch_size] = prior_param[0]
                prior_params[1, i:i + batch_size] = prior_param[1]
                rec_params[0, i:i + batch_size] = rec_param[0]
                rec_params[1, i:i + batch_size] = rec_param[1]
                i += batch_size
    return means, covs, prior_params, rec_params, nlls


def get_predictions_direct(model, val_dl):
    dataset_size = val_dl.dataset.x.shape[0]
    out_features = val_dl.dataset.x.shape[1]

    means = torch.zeros((dataset_size, out_features))
    l_tris = torch.zeros(
        (dataset_size, out_features * (out_features + 1) // 2)
    )
    x_exists = torch.zeros((dataset_size, out_features), dtype=torch.bool)
    i = 0
    model.eval()
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for x, y, z, x_exist in pbar:
                mean, l_tri_flat = model(y, z)
                batch_size = mean.shape[0]
                means[i:i + batch_size] = mean
                l_tris[i:i + batch_size] = l_tri_flat
                x_exists[i:i + batch_size] = x_exist
                i += batch_size
    return means, l_tris, x_exists


def reconstruct_cov(l_tri_flat, n_features, min_d):
    device = l_tri_flat.device
    l_tri = torch.zeros(
        l_tri_flat.shape[0], n_features, n_features, device=device
    )
    ti = torch.tril_indices(n_features, n_features, 0, device=device)
    l_tri[:, ti[0], ti[1]] = l_tri_flat
    identity = torch.eye(n_features, device=device)
    return torch.bmm(l_tri, l_tri.transpose(1, 2)) + identity * min_d, l_tri


def get_nll(
        x, mean, l_tri_flat, min_d, l_tri_loss, message='cvae',
        covariance_matrix=None
):
    n_features = mean.shape[-1]
    if covariance_matrix is not None:
        cov, l_tri = covariance_matrix, None
    else:
        cov, l_tri = reconstruct_cov(l_tri_flat, n_features, 0)
    min_ds = np.logspace(
        np.log10(min_d), -1, int(-np.log10(min_d))
    )
    min_ds = np.concatenate((min_ds, np.array([10 ** i for i in range(10)])))
    identity = torch.eye(n_features, device=x.device)
    caught_exception = False
    i = 0
    for d in min_ds:
        i += 1
        cov_min_d = cov + identity * d
        try:
            if l_tri_loss:
                p = torch.distributions.MultivariateNormal(
                    mean, scale_tril=l_tri
                )
            else:
                # print(mean.shape, cov_min_d.shape)
                p = torch.distributions.MultivariateNormal(
                    mean, covariance_matrix=cov_min_d
                )
            # if caught_exception:
                # print(message, d, 'exception!')
            return -p.log_prob(x), False
        except Exception as e:
            # print(e)
            caught_exception = True
            continue
    print('skip update!')
    # print(cov_min_d[0])
    return None, True


def get_cvae_loss(
        x, mean, l_tri_flat, prior_params, rec_params, samples, n_samples,
        min_d, is_samples=0, mask=None, n_features=None, dont_reduce=False,
):
    n_samples = is_samples if is_samples else n_samples
    n_features = n_features if n_features else mean.shape[-1]
    x = x.repeat(1, n_samples).reshape(
        -1, x.shape[1]
    )
    batch_size = mean.shape[1]
    l_tri_flat_batch_view = l_tri_flat.transpose(0, 1).reshape(
        n_samples * batch_size, -1
    )

    mean_batch_view = mean.transpose(0, 1).reshape(
        n_samples * batch_size, -1
    )
    prior_dist = torch.distributions.Normal(
        prior_params[0], prior_params[1]
    )
    rec_dist = torch.distributions.Normal(
        rec_params[0], rec_params[1]
    )
    kl = torch.distributions.kl.kl_divergence(
        rec_dist, prior_dist
    ).sum(dim=1, keepdim=True)
    # print(rec_params[0].shape, kl.shape)
    if mask is not None:
        # print(l_tri_flat_batch_view.shape)
        cov_batch_view = reconstruct_cov(
            l_tri_flat_batch_view, n_features, 0.
        )[0]
        cov_batch_view = cov_batch_view[:, mask, :][:, :, mask]
        nll, skip_update = get_nll(
            x, mean_batch_view, l_tri_flat_batch_view, min_d, False,
            covariance_matrix=cov_batch_view
        )
        if skip_update:
            return None, skip_update
        else:
            nll = nll.reshape(batch_size, n_samples)  # .mean(dim=1, keepdim=True)
            if dont_reduce:
                return nll, skip_update
            nll = torch.logsumexp(-nll, dim=1, keepdim=True) - np.log(n_samples)
            return torch.cat([-nll, kl], dim=1), skip_update
    else:
        nll = get_nll(
            x, mean_batch_view, l_tri_flat_batch_view, min_d, False,
        )[0].reshape(batch_size, n_samples)  # .mean(dim=1, keepdim=True)
        if dont_reduce:
            return nll
        nll = torch.logsumexp(-nll, dim=1, keepdim=True) - np.log(n_samples)
    if is_samples > 0:
        samples = samples.reshape(is_samples, batch_size, -1)
        prior_ll = prior_dist.log_prob(samples).sum(dim=2)
        rec_ll = rec_dist.log_prob(samples).sum(dim=2)
        # print((nll.squeeze(-1) + prior_ll - rec_ll).shape)
        is_estimate = np.log(is_samples) - torch.logsumexp(
            (- nll.squeeze(-1) + prior_ll - rec_ll).transpose(0, 1), dim=1,
            keepdim=True
        )
        is_estimate.unsqueeze(1)
        # print(is_estimate.mean(dim=0))
        return torch.cat([nll, kl, is_estimate], dim=1)

    return torch.cat([-nll, kl], dim=1)


def get_stats(val_dl, mean, l_tri, min_d, compute_nll=True, covs=None):
    masks = marginalize_non_existing_filters(val_dl.dataset.x_exist)
    x, y, z = val_dl.dataset.x, val_dl.dataset.y, val_dl.dataset
    n_features = mean.shape[1]
    mses = torch.zeros((n_features, 2))
    mses_dist = []
    nlls = [] if compute_nll else None
    conf_acc = []
    cov_mean = torch.zeros((n_features, n_features))
    covs_list = [] if covs is None else None
    for i in range(len(masks)):
        marg_mask = masks[i][0]
        idx_mask = masks[i][1]
        x_marg = x[idx_mask][:, marg_mask]
        mean_marg = mean[idx_mask][:, marg_mask]
        # y_filtered = y[idx_mask]
        # z_filtered = z[idx_mask]
        if covs is not None:
            cov = covs[idx_mask]
        else:
            l_flat_filtered = l_tri[idx_mask]
            cov, _ = reconstruct_cov(l_flat_filtered, n_features, min_d)
            covs_list.append(cov)
        cov_mean += cov.sum(dim=0)
        mse = mse_loss(mean_marg, x_marg, reduction='none')
        mses[marg_mask, 0] += mse.sum(dim=0)
        mses[marg_mask, 1] += mean_marg.shape[0]
        mses_dist.append(mse.mean(dim=1))
        cov = cov[:, marg_mask, :][:, :, marg_mask]
        conf_acc.append(get_confidence_accuracies(
            mean_marg, cov, x_marg
        ))
        if compute_nll:
            nlls.append(get_nll(
                x_marg, mean_marg, None, min_d, False,
                covariance_matrix=cov - torch.eye(cov.shape[1]) * min_d
            )[0])
    mses = mses[:, 0] / mses[:, 1]
    mses_dist = torch.cat(mses_dist, dim=0)
    conf_acc = torch.cat(conf_acc, dim=0)
    nlls = torch.cat(nlls, dim=0) if compute_nll else None
    cov_mean = cov_mean / x.shape[0]
    covs_list = torch.cat(covs_list, dim=0) if covs is None else None
    return mses, mses_dist, nlls, conf_acc, cov_mean, covs_list


cvae_stats = {
    'loss': get_cvae_loss,
}


def get_cvae_stats(model, val_dl, min_d, n_samples=100):
    model.eval()
    n_stats = 3
    results = [[] for _ in range(n_stats)]
    j = 0
    n_features = val_dl.dataset.x.shape[-1]
    maes = torch.zeros((n_features, 2))
    maes_dist = []
    nlls = []
    conf_acc = []
    cov_mean = torch.zeros((n_features, n_features))
    covs = []
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for x, y, z, x_exist in pbar:
                masks = marginalize_non_existing_filters(x_exist)
                mean, l_tri_flat, prior_params, rec_params, samples = model(
                    x, y, z, n_samples=n_samples, ret_latent_samples=True,
                    sample_from_prior=True
                )
                for i in range(len(masks)):
                    marg_mask = masks[i][0]
                    idx_mask = masks[i][1]
                    x_marg = x[idx_mask][:, marg_mask]
                    mean_marg = mean[:, idx_mask][:, :, marg_mask]
                    l_flat_filtered = l_tri_flat[:, idx_mask]
                    y_filtered = y[idx_mask]
                    z_filtered = z[idx_mask]
                    prior_params_filtered = (
                        prior_params[0][idx_mask], prior_params[1][idx_mask]
                    )
                    rec_params_filtered = (
                        rec_params[0][idx_mask], rec_params[1][idx_mask]
                    )
                    cov, _ = reconstruct_cov(l_flat_filtered, n_features, min_d)
                    cov_mean += cov.sum(dim=0)
                    covs.append(cov)
                    cov = cov[:, marg_mask, :][:, :, marg_mask]
                    mae = mse_loss(mean_marg, x_marg, reduction='none')
                    maes[marg_mask, 0] += mae.sum(dim=0)
                    maes[marg_mask, 1] += mean_marg.shape[0]
                    maes_dist.append(mae.mean(dim=1))

                    conf_acc.append(eval.models.get_confidence_accuracies(
                        mean_marg, cov, x_marg
                    ))
                    nlls.append(eval.models.get_nll(
                        x_marg, mean_marg, None, args.min_d, False,
                        covariance_matrix=cov - torch.eye(
                            cov.shape[1]) * args.min_d
                    )[0])

                # j += 1
                # if j % 100 == 0:
                #     break
                mean, l_tri_flat, prior_params, rec_params, samples = model(
                    x, y, z, n_samples=n_samples, ret_latent_samples=True
                )
                mean_batch_view = mean.transpose(0, 1).reshape(
                    n_samples * x.shape[0], -1
                )
                l_tri_batch_view = l_tri_flat.transpose(0, 1).reshape(
                    n_samples * x.shape[0], -1
                )
                cov = reconstruct_cov(
                    l_tri_batch_view, mean.shape[-1], min_d
                )[0]
                x_repeated = x.repeat(1, n_samples).reshape(
                    -1, x.shape[1]
                )
                results[0].append(mean_batch_view)
                mahalanobis_dist = get_mahalanobis_distance(
                    mean_batch_view, cov, x_repeated
                )
                results[1].append(mahalanobis_dist)
                results[2].append(get_cvae_loss(
                    x, mean, l_tri_flat, prior_params, rec_params, samples,
                    n_samples, min_d
                ))
                cov_sum += cov.sum(dim=0)

    final_res = []
    for i in range(n_stats):
        final_res.append(torch.cat(results[i], dim=0))
    final_res.append(cov_sum / (val_dl.dataset.x.shape[0] * n_samples))
    return final_res


"""
def get_cvae_stats(model, val_dl, stats):
    model.eval()
    results = [[] for _ in range(len(stats))]
    j = 0
    with tqdm(val_dl) as pbar:
        pbar.set_description("Running the model")
        with torch.no_grad():
            for x, y, z in pbar:
                j += 1
                # if j % 100 == 0:
                # break
                for i, stat in enumerate(stats):
                    results[i].append(
                        cvae_stats[stat[0]](model, x, y, z, **stat[1])
                    )
    results_dict = {}
    for i, stat in enumerate(stats):
        results_dict[stat[0]] = torch.cat(results[i], dim=0)
    return results_dict
"""
