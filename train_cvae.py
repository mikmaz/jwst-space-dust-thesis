from trainer.cvae import CVAETrainer
import utils
from ast import literal_eval
import dataset
from torch.utils.data import DataLoader
from cvae_utils import make_net, parse_args
from models.cvae import CVAE


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
        args.normalize
    )
    direct_net = make_net(args.direct_net, args.direct_kwargs)
    prior_net = make_net(args.prior_net, args.prior_kwargs)
    rec_net = make_net(args.recognition_net, args.recognition_kwargs)
    gen_net = make_net(args.gen_net, args.gen_kwargs)
    model = CVAE(
        direct_net, prior_net, rec_net, gen_net, args.recurrent_connection
    )
    print(model)
    trainer = CVAETrainer(
        train_dl, val_dl, model, device, args.lr, args.min_d, args.l_tri_loss,
        args.gsnn_frac, args.n_latent_samples, args.lr_decay, args.weight_decay
    )
    trainer.train(args.n_epochs, args.stats_path, args.grad_clip)


if __name__ == "__main__":
    main(parse_args())
