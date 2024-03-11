from ast import literal_eval

import torch

import dataset
import train_utils
import utils
from trainer.direct import DirectTrainer


def main(args):
    print(args)
    device = utils.get_device()
    print("Device:", device)
    utils.enforce_reproducibility()
    train_dl, val_dl = dataset.get_data_loaders(
        f'{args.data_path}/sampled_filters_train.pkl',
        f'{args.data_path}/sampled_filters_val.pkl',
        utils.none_literal_eval(args.drop_filters),
        utils.none_literal_eval(args.drop_dust_features),
        args.batch,
        args.n_workers,
        args.normalize,
        args.swap_features
    )

    model_kwargs = literal_eval(args.model_kwargs)
    model = utils.models_dict[args.model](**model_kwargs).to(device)
    if args.load_model is not None:
        model.load_state_dict(
            torch.load(
                f'{args.stats_path}/old/checkpoint/{args.load_model}',
                map_location=device
            )
        )
    print(model)
    trainer = DirectTrainer(
        train_dl, val_dl, model, device, args.lr, args.min_d, args.l_tri_loss,
        args.lr_decay, args.weight_decay
    )
    trainer.train(args.n_epochs, args.stats_path, args.grad_clip)


if __name__ == "__main__":
    main(train_utils.parse_args())
