import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import utils
from ast import literal_eval
import torch.nn.functional as F
import math
import numpy as np
import dataset
import train_utils


def loss_f(mean, l_flat, actual_value, min_d, l_tri_loss=False):
    batch_size, n_features = actual_value.shape
    cov, l_tri = utils.reconstruct_cov(l_flat, n_features, 0)
    min_ds = np.logspace(np.log10(min_d), -1, int(-np.log10(min_d)))
    identity = torch.eye(n_features, device=actual_value.device)
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
                print(i, 'exception!')
            return -p.log_prob(actual_value).mean(), False
        except Exception as _:
            caught_exception = True
            continue
    print('skip update!')
    return None, True


def eval_batch(model, x, y, z, device, model_type, min_d, l_tri_loss=False):
    x = x.to(device=device)
    y = y.to(device=device)
    z = z.to(device=device)
    model_in = y if model_type == 'likelihood' else x
    true_out = x if model_type == 'likelihood' else y
    mean, l_tri = model(model_in, z)
    loss, skip_update = loss_f(mean, l_tri, true_out, min_d, l_tri_loss)
    with torch.no_grad():
        mae = F.l1_loss(mean, true_out)

    return loss, mae, skip_update


def evaluate(model, dl, device, model_type, min_d, l_tri_loss=False):
    val_losses = []
    val_mses = []
    model.eval()
    with torch.no_grad():
        i = 0
        for x, y, z in dl:
            i += 1
            # if i % 100 == 0:
            # print(i, 'eval', loss)
            # break
            loss, mse, skip_update = eval_batch(
                model, x, y, z, device, model_type, min_d, l_tri_loss
            )
            if not skip_update:
                val_mses.append(mse.item())
                val_losses.append(loss.item())
    mean_loss = sum(val_losses[:-1]) / len(val_losses[:-1])
    mean_mse = sum(val_mses[:-1]) / len(val_mses[:-1])
    return mean_loss, mean_mse


def train(
        model,
        train_dl,
        val_dl,
        optimizer,
        n_epochs,
        device,
        stats_path,
        model_type,
        min_d=0.0,
        grad_clip=0,
        scheduler=None,
        l_tri_loss=False
):
    losses = []
    mses = []
    with open(f'{stats_path}/train_losses.txt', 'w+') as f:
        f.write("Training losses:\n")
    with open(f'{stats_path}/val_losses.txt', 'w+') as f:
        f.write("Validation losses:\n")
    with open(f'{stats_path}/train_mses.txt', 'w+') as f:
        f.write("Training MSEs:\n")
    with open(f'{stats_path}/val_mses.txt', 'w+') as f:
        f.write("Validation MSEs:\n")
    best_val_loss = None
    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            epoch_losses = []
            epoch_mses = []
            i = 0
            for x, y, z in train_dl:
                i += 1
                # if i % 100 == 0:
                   #  print(i, loss)
                # break
                model.train()
                optimizer.zero_grad()
                loss, mse, skip_update = eval_batch(
                    model, x, y, z, device, model_type, min_d, l_tri_loss
                )
                if not skip_update:
                    epoch_mses.append(mse.item())
                    epoch_losses.append(loss.item())
                    loss.backward()
                    # torch.nn.utils.clip_grad_value_(model.parameters(), 1e5)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            grad_clip
                        )
                    optimizer.step()

            if (epoch + 1) % max(1, (n_epochs // 5)) == 0:
                torch.save(
                    model.state_dict(),
                    f"{stats_path}/checkpoint/model_{epoch}.pt"
                )
                torch.save(
                    optimizer.state_dict(),
                    f"{stats_path}/checkpoint/optim_{epoch}.pt"
                )

            print(f'Epoch: {epoch}, train loss: ' +
                  f'{sum(epoch_losses[:-1]) / len(epoch_losses[:-1])}')
            with open(f'{stats_path}/train_losses.txt', 'a+') as f:
                f.write(f'{sum(epoch_losses[:-1]) / len(epoch_losses[:-1])}\n')
            with open(f'{stats_path}/train_mses.txt', 'a+') as f:
                f.write(f'{sum(epoch_mses[:-1]) / len(epoch_mses[:-1])}\n')

            val_loss, val_mse = evaluate(
                model, val_dl, device, model_type, min_d, l_tri_loss
            )
            if scheduler is not None:
                scheduler.step(val_mse)
            with open(f'{stats_path}/val_losses.txt', 'a+') as f:
                f.write(f'{val_loss}\n')
            with open(f'{stats_path}/val_mses.txt', 'a+') as f:
                f.write(f'{val_mse}\n')
            if best_val_loss is None or best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"{stats_path}/checkpoint/best_model.pt"
                )
            losses.append(sum(epoch_losses[:-1]) / len(epoch_losses[:-1]))
            mses.append(sum(epoch_mses[:-1]) / len(epoch_mses[:-1]))

    return model, losses


def main(args):
    print(args)
    device = utils.get_device()
    print("Device:", device)
    utils.enforce_reproducibility()
    drop_filters = literal_eval(args.drop_filters) \
        if args.drop_filters is not None else None
    drop_dust_features = literal_eval(args.drop_dust_features) \
        if args.drop_dust_features is not None else None
    train_dataset = dataset.get_dataset_from_file(
        f'{args.data_path}/sampled_filters_train.pkl',
        drop_filters,
        drop_dust_features
    )
    val_dataset = dataset.get_dataset_from_file(
        f'{args.data_path}/sampled_filters_val.pkl',
        drop_filters,
        drop_dust_features
    )
    if args.normalize:
        x_stats, y_stats, z_stats = train_dataset.normalize()
        val_dataset.normalize(x_stats, y_stats, z_stats)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch,
        num_workers=args.n_workers,
        shuffle=True
    )
    val_dl = DataLoader(
        val_dataset, batch_size=args.batch, num_workers=args.n_workers
    )

    model_kwargs = literal_eval(args.model_kwargs)
    model = utils.models_dict[args.model](**model_kwargs).to(device)
    print(model)
    lr = args.lr
    optimizer = Adam(model.parameters(), lr=lr)
    if args.lr_decay > 0:
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=args.lr_decay, threshold=1e-3, patience=10
        )
    else:
        scheduler = None
    epochs = args.n_epochs
    model, losses = train(
        model,
        train_dl,
        val_dl,
        optimizer,
        epochs,
        device,
        args.stats_path,
        args.model_type,
        args.min_d,
        args.grad_clip,
        scheduler=scheduler,
        l_tri_loss=args.l_tri_loss
    )
    torch.save(
        model.state_dict(), f"{args.stats_path}/checkpoint/final_model.pt"
    )


if __name__ == "__main__":
    main(train_utils.parse_args())
