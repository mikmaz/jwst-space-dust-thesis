import numpy as np
import torch
import torch.nn.functional as F
from trainer.trainer import Trainer


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
        x, mean, l_tri_flat, device, min_d, l_tri_loss, message='cvae',
        marg_mask=None
):
    n_features = mean.shape[-1]
    cov, l_tri = reconstruct_cov(
        l_tri_flat, n_features if marg_mask is None else marg_mask.shape[0], 0
    )
    if marg_mask is not None:
        cov = cov[:, marg_mask, :][:, :, marg_mask]
    min_ds = np.logspace(
        np.log10(min_d), -1, int(-np.log10(min_d))
    )
    identity = torch.eye(n_features, device=device)
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
                p = torch.distributions.MultivariateNormal(
                    mean, covariance_matrix=cov_min_d
                )
            if caught_exception:
                print(message, i, 'exception!')
            return -p.log_prob(x), False
        except Exception as _:
            caught_exception = True
            continue
    print('skip update!')
    return None, True


def marginalize_non_existing_filters(x_exist):
    def binary(x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0)

    n_features = x_exist.shape[1]
    binary_nums = torch.tensor(
        [2**i for i in range(n_features)], device=x_exist.device
    )
    keys = (x_exist * binary_nums).sum(dim=1)
    unique, inverse_idx = torch.unique(keys, return_inverse=True)
    idx = []
    for i in range(unique.shape[0]):
        idx.append((binary(unique[i], n_features), inverse_idx == i))
    return idx


class DirectTrainer(Trainer):
    def __init__(
            self, train_dl, val_dl, model, device, lr, min_d,
            l_tri_loss, lr_decay=0, weight_decay=0
    ):
        super(DirectTrainer, self).__init__(
            train_dl, val_dl, model, device, lr, lr_decay, weight_decay
        )
        self.min_d = min_d
        self.l_tri_loss = l_tri_loss

    def loss_f(self, x, y, z, mean, l_flat, marg_mask, min_d, l_tri_loss):
        return get_nll(
            x, mean, l_flat, self.device, min_d, l_tri_loss, marg_mask=marg_mask
        )

    def mae_f(self, x, mean):
        with torch.no_grad():
            mae = F.l1_loss(mean, x, reduction='none').mean(dim=1)

        return mae

    def eval_batch(self, x, y, z, x_exist):
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        z = z.to(device=self.device)
        x_exist = x_exist.to(device=self.device)
        mean, l_flat = self.model(y, z)
        masks = marginalize_non_existing_filters(x_exist)
        loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        mae = torch.tensor(0, dtype=torch.float32, device=self.device)
        skip_update = False
        for i in range(len(masks)):
            marg_mask = masks[i][0]
            idx_mask = masks[i][1]
            x_marg = x[idx_mask][:, marg_mask]
            mean_marg = mean[idx_mask][:, marg_mask]
            l_flat_filtered = l_flat[idx_mask]
            y_filtered = y[idx_mask]
            z_filtered = z[idx_mask]
            nll, skip_update = self.loss_f(
                x_marg, y_filtered, z_filtered, mean_marg, l_flat_filtered,
                marg_mask, self.min_d, self.l_tri_loss
            )
            mae_marg = self.mae_f(x_marg, mean_marg)
            if skip_update:
                return None, None, skip_update
            loss += nll.sum()
            mae += mae_marg.sum()

        batch_size = x.shape[0]
        return loss / batch_size, mae / batch_size, skip_update
