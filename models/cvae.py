import torch
import torch.nn as nn


class CVAE(nn.Module):
    def __init__(
            self, direct_net, prior_net, recognition_net, gen_net,
            recurrent_connection
    ):
        super(CVAE, self).__init__()
        self.direct_net = direct_net
        self.prior = prior_net
        self.recognition_net = recognition_net
        self.gen_net = gen_net
        self.recurrent_connection = recurrent_connection
        self.latent_size = recognition_net.out_features

    def forward(
            self, x, y, z, n_samples=1, sample_from_prior=False,
            ret_latent_samples=False
    ):
        mean_direct, l_tri_direct = self.direct_net(y, z, skip_last_layer=True)
        if self.recurrent_connection:
            mean_hat_direct, _ = self.direct_net.last_layer(
                mean_direct, l_tri_direct
            )
            prior_in = torch.cat([mean_hat_direct, y], dim=1)
        else:
            prior_in = y
        prior_mean, prior_scale = self.prior(prior_in, z)
        rec_mean, rec_scale = self.recognition_net(torch.cat([x, y], dim=1), z)
        if sample_from_prior:
            dist = torch.distributions.Normal(prior_mean, prior_scale)
        else:
            dist = torch.distributions.Normal(rec_mean, rec_scale)

        batch_size = x.shape[0]
        samples = dist.rsample((n_samples,)).reshape(batch_size * n_samples, -1)
        z_repeated = z.repeat(n_samples, 1)
        gen_mean, gen_l_tri = self.gen_net(
            samples, z_repeated, skip_last_layer=True
        )

        gen_mean = gen_mean + mean_direct.repeat(n_samples, 1)
        gen_l_tri = gen_l_tri + l_tri_direct.repeat(n_samples, 1)
        mean, l_tri_flat = self.direct_net.last_layer(gen_mean, gen_l_tri)

        mean = mean.reshape(n_samples, batch_size, -1)
        l_tri_flat = l_tri_flat.reshape(n_samples, batch_size, -1)
        prior_dist = (prior_mean, prior_scale)
        rec_dist = (rec_mean, rec_scale)
        if ret_latent_samples:
            return mean, l_tri_flat, prior_dist, rec_dist, samples

        return mean, l_tri_flat, prior_dist, rec_dist
