import torch.nn as nn


class BasicBlockConv(nn.Module):
    def __init__(
            self,
            in_channels,
            kernel_size,
            scale_channels=None
    ):
        super(BasicBlockConv, self).__init__()
        block_channels = in_channels \
            if scale_channels is None else scale_channels
        padding = kernel_size // 2
        scaler = nn.Sequential(
            nn.Conv1d(
                in_channels,
                block_channels,
                kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm1d(block_channels)
        ) if scale_channels else None
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                block_channels,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(block_channels),
            nn.ReLU(),
            nn.Conv1d(
                block_channels,
                block_channels,
                kernel_size,
                padding=padding
            ),
            nn.BatchNorm1d(block_channels)
        )
        self.final_relu = nn.ReLU()
        self.scaler = scaler

    def forward(self, x):
        identity = x
        x = self.net(x)
        if self.scaler is not None:
            # print(identity.shape)
            identity = self.scaler(identity)
        # print(x.shape, identity.shape)
        return self.final_relu(x + identity)


class BasicBlockLinear(nn.Module):
    def __init__(self, in_features, scale_features=None):
        super(BasicBlockLinear, self).__init__()
        scaler = nn.Sequential(
            nn.Linear(in_features, scale_features, bias=False),
            nn.BatchNorm1d(scale_features)
        ) if scale_features else None
        block_features = in_features if scaler is None else scale_features
        self.net = nn.Sequential(
            nn.Linear(in_features, block_features),
            nn.BatchNorm1d(block_features),
            nn.ReLU(),
            nn.Linear(block_features, block_features),
            nn.BatchNorm1d(block_features)
        )
        self.final_relu = nn.ReLU()
        self.scaler = scaler

    def forward(self, x):
        identity = x
        x = self.net(x)
        if self.scaler is not None:
            identity = self.scaler(identity)
        return self.final_relu(x + identity)


class ResNet(nn.Module):
    def __init__(
            self,
            in_features,
            n_blocks,
            scale_features=None,
            kernel_size=7,
            linear=False
    ):
        super(ResNet, self).__init__()
        blocks = [
            BasicBlockLinear(in_features, scale_features)
            if linear else
            BasicBlockConv(
                in_features, kernel_size, scale_features
            )
        ]
        block_features = in_features \
            if scale_features is None else scale_features
        blocks += [
            BasicBlockLinear(block_features)
            if linear else
            BasicBlockConv(
                block_features, kernel_size,
            )
            for _ in range(n_blocks - 1)
        ]
        self.res_layers = nn.ModuleList(blocks)

    def forward(self, x):
        for res_layer in self.res_layers:
            x = res_layer(x)
        return x
