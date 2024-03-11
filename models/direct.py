import torch.nn as nn
import torch
from models.resnet import ResNet


class DirectLikelihoodResNet(nn.Module):
    def __init__(
            self,
            in_features=8,
            out_features=33,
            up_dims=(33, 33, 33),
            n_blocks=(1, 1, 1),
            beta=0.1,
            min_diag=0.,
            kernel_size=None,
    ):
        super(DirectLikelihoodResNet, self).__init__()
        self.common_res_net = ResNet(
            in_features,
            n_blocks[0],
            scale_features=up_dims[0]
            if up_dims[0] != in_features else None,
            linear=True
        )

        self.mean_res_net = ResNet(
            up_dims[0],
            n_blocks[1],
            scale_features=up_dims[1] if up_dims[0] != up_dims[1] else None,
            linear=True
        )
        self.mean_fc_net = nn.Linear(up_dims[1], out_features)
        if kernel_size is None:
            self.cov_res_net = ResNet(
                up_dims[0],
                n_blocks[2],
                scale_features=up_dims[2]
                if up_dims[0] != up_dims[2] else None,
                linear=True
            )
            cov_final_size = up_dims[2]
        else:
            self.cov_res_net = nn.Sequential(
                nn.Unflatten(1, (1, up_dims[0])),
                ResNet(
                    1,
                    n_blocks[2],
                    scale_features=up_dims[2],
                    linear=False,
                    kernel_size=kernel_size
                ),
                nn.Flatten(1)
            )
            cov_final_size = up_dims[2] * up_dims[0]
        self.cov_fc_net = nn.Linear(
            cov_final_size,
            out_features * (out_features + 1) // 2
        )

        self.out_features = out_features
        diag_indices = [0]
        for i in range(out_features - 1):
            diag_indices.append(diag_indices[-1] + i + 2)
        self.register_buffer('diag_indices', torch.tensor(diag_indices))
        self.register_buffer('min_diag', torch.tensor(min_diag))
        self.register_buffer('beta', torch.tensor(beta))

    def last_layer(self, mean, l_tri):
        mean = self.mean_fc_net(mean)
        l_flat = self.cov_fc_net(l_tri)

        l_flat[:, self.diag_indices] = nn.functional.softplus(
            l_flat[:, self.diag_indices], beta=self.beta
        ) + self.min_diag
        # l_flat[:, self.diag_indices] = torch.exp(
        #     l_flat[:, self.diag_indices]
        # ) + self.min_diag
        return mean, l_flat

    def forward(self, y, z, skip_last_layer=False):
        x = torch.cat((y, z), dim=1)
        x = self.common_res_net(x)
        if skip_last_layer:
            return self.mean_res_net(x), self.cov_res_net(x)
        else:
            l_flat = self.cov_fc_net(self.cov_res_net(x))

            l_flat[:, self.diag_indices] = nn.functional.softplus(
                l_flat[:, self.diag_indices], beta=self.beta
            ) + self.min_diag
            # l_flat[:, self.diag_indices] = torch.exp(
            #     l_flat[:, self.diag_indices]
            # ) + self.min_diag
            return self.mean_fc_net(self.mean_res_net(x)), l_flat


class DirectUnivariateLikelihoodResNet(nn.Module):
    def __init__(
            self,
            in_features=8,
            out_features=33,
            up_dims=(33, 33, 33),
            n_blocks=(1, 1, 1),
            beta=0.1,
            min_var=0.,
            kernel_size=None
    ):
        super(DirectUnivariateLikelihoodResNet, self).__init__()
        self.common_res_net = ResNet(
            in_features,
            n_blocks[0],
            scale_features=up_dims[0]
            if up_dims[0] != in_features else None,
            linear=True
        )

        self.mean_res_net = ResNet(
            up_dims[0],
            n_blocks[1],
            scale_features=up_dims[1] if up_dims[0] != up_dims[1] else None,
            linear=True
        )
        self.mean_fc_net = nn.Linear(up_dims[1], out_features)
        if kernel_size is None:
            self.scale_res_net = ResNet(
                up_dims[0],
                n_blocks[2],
                scale_features=up_dims[2]
                if up_dims[0] != up_dims[2] else None,
                linear=True
            )
            scale_final_size = up_dims[2]
        else:
            self.scale_res_net = nn.Sequential(
                nn.Unflatten(1, (1, up_dims[0])),
                ResNet(
                    1,
                    n_blocks[2],
                    scale_features=up_dims[2],
                    linear=False,
                    kernel_size=kernel_size
                ),
                nn.Flatten(1)
            )
            scale_final_size = up_dims[2] * up_dims[0]
        self.scale_fc_net = nn.Linear(
            scale_final_size,
            out_features
        )

        self.out_features = out_features
        diag_indices = [0]
        for i in range(out_features - 1):
            diag_indices.append(diag_indices[-1] + i + 2)
        self.register_buffer('min_var', torch.tensor(min_var))
        self.register_buffer('beta', torch.tensor(beta))

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x = self.common_res_net(x)
        mean = self.mean_fc_net(self.mean_res_net(x))
        scale = nn.functional.softplus(
            self.scale_fc_net(self.scale_res_net(x)), beta=self.beta
        ) + self.min_var
        return mean, scale


class FixedPrior:
    def __init__(self, latent_size):
        self.latent_size = latent_size

    def __call__(self, y, z):
        batch_size = y.shape[0]
        device = y.device
        prior_mean = torch.zeros(
            (batch_size, self.latent_size), device=device
        )
        prior_scale = torch.ones(
            (batch_size, self.latent_size), device=device
        )
        return prior_mean, prior_scale


def generate_dnn(in_features, hidden_neurons, out_features=None):
    if len(hidden_neurons) == 0:
        return nn.Linear(in_features, out_features)
    layers = [
        nn.Linear(in_features, hidden_neurons[0]),
        nn.BatchNorm1d(hidden_neurons[0]),
        nn.ReLU()
    ]
    for i in range(len(hidden_neurons) - 1):
        layers += [
            nn.Linear(hidden_neurons[i], hidden_neurons[i + 1]),
            nn.BatchNorm1d(hidden_neurons[i + 1]),
            nn.ReLU()
        ]
    if out_features is not None:
        layers += [nn.Linear(hidden_neurons[-1], out_features)]
    return nn.Sequential(*layers)


class SimpleDirectLikelihoodNet(nn.Module):
    def __init__(
            self,
            in_features=8,
            out_features=33,
            common_hidden_neurons=(33,),
            mean_hidden_neurons=(),
            cov_hidden_neurons=(),
            beta=0.1,
            min_diag=0.
    ):
        super(SimpleDirectLikelihoodNet, self).__init__()
        self.common_net = generate_dnn(in_features, common_hidden_neurons)
        self.mean_net = generate_dnn(
            common_hidden_neurons[-1], mean_hidden_neurons, out_features
        )
        self.cov_net = generate_dnn(
            common_hidden_neurons[-1],
            cov_hidden_neurons,
            out_features * (out_features + 1) // 2
        )
        self.out_features = out_features
        diag_indices = [0]
        for i in range(out_features - 1):
            diag_indices.append(diag_indices[-1] + i + 2)
        self.register_buffer('diag_indices', torch.tensor(diag_indices))
        self.register_buffer('min_diag', torch.tensor(min_diag))
        self.register_buffer('beta', torch.tensor(beta))

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)
        x = self.common_net(x)
        mean = self.mean_net(x)

        l_flat = self.cov_net(x)

        l_flat[:, self.diag_indices] = nn.functional.softplus(
            l_flat[:, self.diag_indices], beta=self.beta
        ) + self.min_diag
        # l_flat[:, self.diag_indices] = torch.exp(
        #     l_flat[:, self.diag_indices]
        # ) + self.min_diag
        return mean, l_flat
