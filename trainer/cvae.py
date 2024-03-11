import torch
import torch.nn.functional as F
from trainer.trainer import Trainer
from trainer.direct import get_nll, marginalize_non_existing_filters


class CVAETrainer(Trainer):
    def __init__(
            self, train_dl, val_dl, model, device, lr, min_d,
            l_tri_loss, gsnn_frac, n_samples, lr_decay=0, weight_decay=0,
    ):
        super(CVAETrainer, self).__init__(
            train_dl, val_dl, model, device, lr, lr_decay, weight_decay
        )
        self.min_d = min_d
        self.l_tri_loss = l_tri_loss
        self.gsnn_frac = gsnn_frac
        self.n_samples = n_samples

    def get_cvae_nll(self, x, mean, l_tri_flat, marg_mask, message='cvae'):
        n_samples, batch_size, n_features = mean.shape
        mean_batch_view = mean.transpose(0, 1).reshape(
            n_samples * batch_size, -1
        )
        l_tri_flat_batch_view = l_tri_flat.transpose(0, 1).reshape(
            n_samples * batch_size, -1
        )
        x_repeated = x.repeat(1, n_samples).reshape(
            -1, x.shape[1]
        )
        nll = get_nll(
            x_repeated, mean_batch_view, l_tri_flat_batch_view, self.device,
            self.min_d, self.l_tri_loss, message=message, marg_mask=marg_mask
        )
        # print(nll[0])
        return nll

    def loss_f(
            self, x, y, z, mean, l_tri_flat, prior_params, rec_params, marg_mask
    ):
        cvae_nll, skip_update = self.get_cvae_nll(
            x, mean, l_tri_flat, marg_mask
        )
        if skip_update:
            return None, True

        prior_dist = torch.distributions.Normal(
            prior_params[0], prior_params[1]
        )
        rec_dist = torch.distributions.Normal(rec_params[0], rec_params[1])
        kl = torch.distributions.kl.kl_divergence(
            rec_dist, prior_dist
        ).sum(dim=1)
        kl = kl.repeat(self.n_samples, 1).transpose(0, 1).flatten()
        cvae_loss = cvae_nll + kl
        if self.gsnn_frac > 0:
            gsnn_mean, gsnn_l_tri_flat, _, _ = self.model(
                x, y, z, self.n_samples, sample_from_prior=True
            )
            gsnn_nll, skip_update = self.get_cvae_nll(
                x, gsnn_mean, gsnn_l_tri_flat, marg_mask, message='gsnn'
            )
            if skip_update:
                return None, True
            combined_loss = (1 - self.gsnn_frac) * cvae_loss + \
                self.gsnn_frac * gsnn_nll
            return combined_loss, False
        else:
            return cvae_loss, False

    def mae_f(self, x, mean):
        n_samples, batch_size, _ = mean.shape
        mean_batch_view = mean.transpose(0, 1).reshape(
            n_samples * batch_size, -1
        )
        x_repeated = x.repeat(1, n_samples).reshape(
            -1, x.shape[1]
        )
        with torch.no_grad():
            mae = F.l1_loss(
                mean_batch_view, x_repeated, reduction='none'
            ).mean(dim=1)
        return mae

    def eval_batch(self, x, y, z, x_exist):
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        z = z.to(device=self.device)
        x_exist = x_exist.to(device=self.device)
        mean, l_tri_flat, prior_params, rec_params = self.model(
            x, y, z, n_samples=self.n_samples
        )
        masks = marginalize_non_existing_filters(x_exist)
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        mae = torch.tensor(0, dtype=torch.float32, device=self.device)
        skip_update = False
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
            losses, skip_update = self.loss_f(
                x_marg, y_filtered, z_filtered, mean_marg, l_flat_filtered,
                prior_params_filtered, rec_params_filtered, marg_mask
            )
            mae_marg = self.mae_f(x_marg, mean_marg)
            if skip_update:
                return None, None, skip_update
            loss += losses.sum()
            mae += mae_marg.sum()

        div = x.shape[0] * self.n_samples
        return loss / div, mae / div, skip_update
