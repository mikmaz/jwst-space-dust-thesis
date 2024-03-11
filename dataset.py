from torch.utils.data import Dataset
import pickle
import torch
from torch.utils.data import DataLoader

filter_indexes = {
    'F070W': 0, 'F090W': 1, 'F115W': 2, 'F140M': 3, 'F150W': 4, 'F162M': 5,
    'F164N': 6, 'F182M': 7, 'F187N': 8, 'F200W': 9, 'F210M': 10, 'F212N': 11,
    'F250M': 12, 'F277W': 13, 'F300M': 14, 'F322W2': 15, 'F323N': 16,
    'F335M': 17, 'F356W': 18, 'F360M': 19, 'F405N': 20, 'F410M': 21,
    'F430M': 22, 'F444W': 23, 'F460M': 24, 'F466N': 25, 'F470N': 26,
    'F480M': 27, 'F560W': 28, 'F770W': 29, 'F1000W': 30, 'F1130W': 31,
    'F1280W': 32, 'F1500W': 33, 'F1800W': 34, 'F2100W': 35, 'F2550W': 36
}
y_feature_names_full = [
    'dust mass', 'grain size', 'silicate fraction', "dust temperature",
    'clump factor', "supernova luminosity", "supernova temperature"
]

dust_feature_indexes = {
    'mass': 0, 'grain size': 1, 'silicate fraction': 2, 'dust temperature': 3,
    'clump factor': 4, 'luminosity': 5, 'supernova temperature': 6
}

train_data_stats = {
    'x_stats': (
        torch.tensor([
            21.751858, 21.814981, 21.9899, 22.141357, 22.195036, 22.28943,
            56.87975, 22.37718, 22.404573, 22.410437, 22.453005, 22.473066,
            22.527842, 22.509453, 22.51926, 22.418331, 41.75808, 22.488182,
            22.445654, 22.44901, 22.413263, 22.389057, 22.381468, 22.333538,
            22.327894, 51.01546, 25.948057, 22.309793, 22.207935, 21.971094,
            21.731707, 21.848988, 22.037788, 22.26296, 22.332355, 22.55603,
            23.069767
        ]),
        torch.tensor([
            2.9622726, 2.9763088, 2.9877434, 3.0301025, 3.024199, 3.0466042,
            9.250725, 3.056043, 3.0999408, 3.0622647, 3.1085138, 3.1344073,
            3.1824756, 3.2055902, 3.254263, 3.2834396, 17.51483, 3.3159444,
            3.3328629, 3.3618927, 3.4181442, 3.4029803, 3.4278345, 3.4190295,
            3.461694, 16.164518, 11.803844, 3.4629092, 3.5103052, 3.612073,
            3.7003016, 3.7113857, 3.6594374, 3.6622696, 3.7065406, 3.7179275,
            3.9101102
        ])
    ),
    'y_stats': (
        torch.tensor([
            -3.5023894e+00, -8.0205125e-01, 4.9914283e-01, 7.8838416e+02,
            2.4976912e-01, 4.9981022e+00, 8.9802295e+03
        ]),
        torch.tensor([
            1.4388015e+00, 8.6675882e-01, 2.8804469e-01, 4.2021497e+02,
            1.4451878e-01, 7.3062932e-01, 2.8997354e+03
        ])
    ),
    'z_stats': (torch.tensor([0.0075501]), torch.tensor([0.00430225]))
}


def get_idx_to_keep(idx_dict, to_drop):
    idx_to_drop = [idx_dict[feature] for feature in to_drop]
    return [idx for idx in idx_dict.values() if idx not in idx_to_drop]


class JWSTSpaceDustDataset(Dataset):
    def __init__(
            self, x, y, z, x_exist, drop_filters=None, drop_dust_features=None,
            swap_features=False
    ):
        self.x = x
        self.y = y
        self.z = z
        self.x_exist = x_exist
        self.drop_filters = drop_filters
        self.drop_dust_features = drop_dust_features
        self.swap_features = swap_features
        if drop_filters is not None:
            x_idx_to_keep = get_idx_to_keep(filter_indexes, drop_filters)
            self.x = self.x[:, x_idx_to_keep]
            self.x_exist = self.x_exist[:, x_idx_to_keep]
        if drop_dust_features is not None:
            idx_to_keep = get_idx_to_keep(
                dust_feature_indexes, drop_dust_features
            )
            self.y = self.y[:, idx_to_keep]
        self.y_exist = torch.tensor([True] * self.y.shape[1])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.swap_features:
            return self.y[idx], self.x[idx], self.z[idx], self.y_exist
        return self.x[idx], self.y[idx], self.z[idx], self.x_exist[idx]

    def normalize(self, x_stats=None, y_stats=None, z_stats=None):
        def normalize_feature(feature, stats):
            if stats is None:
                mean = feature.mean(dim=0)
                sd = feature.std(dim=0)
            else:
                mean, sd = stats
            return (feature - mean) / sd, mean, sd

        self.x, x_mean, x_sd = normalize_feature(self.x, x_stats)
        self.y, y_mean, y_sd = normalize_feature(self.y, y_stats)
        self.z, z_mean, z_sd = normalize_feature(self.z, z_stats)
        return (x_mean, x_sd), (y_mean, y_sd), (z_mean, z_sd)


def get_dataset_from_file(
        data_path, drop_filters=None, drop_dust_features=None,
        swap_features=False
):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    x = torch.tensor(data['X'])
    y = torch.tensor(data['y'])
    z = torch.tensor(data['z'])
    x_exist = torch.tensor(data['X_exist'])
    return JWSTSpaceDustDataset(
        x, y, z, x_exist, drop_filters, drop_dust_features, swap_features
    )


def filter_stat(stat, stats_to_drop, feature_indexes):
    if stats_to_drop is None:
        return stat
    idx_to_keep = get_idx_to_keep(feature_indexes, stats_to_drop)
    return stat[0][idx_to_keep], stat[1][idx_to_keep]


def get_data_loaders(
        train_data_path=None, val_data_path=None, drop_filters=None,
        drop_dust_features=None, batch_size=64, n_workers=1, normalize=False,
        swap_features=False
):
    train_dataset = None if train_data_path is None else get_dataset_from_file(
        train_data_path,
        drop_filters,
        drop_dust_features,
        swap_features
    )
    val_dataset = None if val_data_path is None else get_dataset_from_file(
        val_data_path,
        drop_filters,
        drop_dust_features,
        swap_features
    )

    if normalize and train_dataset is not None:
        x_stats, y_stats, z_stats = train_dataset.normalize()
        if val_dataset is not None:
            val_dataset.normalize(x_stats, y_stats, z_stats)
    elif normalize and val_dataset is not None:
        val_dataset.normalize(
            filter_stat(
                train_data_stats['x_stats'], drop_filters, filter_indexes
            ),
            filter_stat(
                train_data_stats['y_stats'], drop_dust_features,
                dust_feature_indexes
            ),
            train_data_stats['z_stats']
        )

    train_dl = None if train_dataset is None else DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True
    )
    val_dl = None if val_dataset is None else DataLoader(
        val_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )
    return train_dl, val_dl
