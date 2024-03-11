import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@', description="JWST space dust trainer"
    )
    parser.add_argument(
        "--model", default='direct_likelihood', type=str,
        help="which model to train"
    )
    parser.add_argument(
        "--model_type", default='likelihood', type=str,
        help="whether model predicts likelihood or posterior"
    )
    parser.add_argument(
        "--model_kwargs", default='{}', type=str,
        help="model's keyword arguments passed as a dictionary"
    )
    parser.add_argument(
        "--batch", default=64, type=int, help="batch size"
    )
    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--n_epochs", default=100, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--n_workers", default=4, type=int,
        help="number of workers used in dataloader"
    )
    parser.add_argument(
        "--grad_clip", default=0, type=int, help="gradient clipping"
    )
    parser.add_argument(
        "--min_d", default=0., type=float,
        help="minimal values of diagonal elements of covariance matrix"
    )
    parser.add_argument(
        "--weight_decay", default=0., type=float, help="weight decay"
    )
    parser.add_argument("--lr_decay", default=0., type=float, help="lr decay")
    parser.add_argument(
        "--normalize", action="store_true",
        help="if set, datasets will be normalized"
    )
    parser.add_argument(
        "--l_tri_loss", action="store_true",
        help="if set, loss will be calculated only for lower triangular part"
    )
    parser.add_argument(
        "--swap_features", action="store_true",
        help="swap x and y features in dataset"
    )
    parser.add_argument(
        "--drop_filters", default=None, type=str,
        help="dictionary of filters to be dropped from dataset"
    )
    parser.add_argument(
        "--drop_dust_features", default=None, type=str,
        help="dictionary of spacedust features to be dropped from dataset"
    )
    parser.add_argument(
        "--load_model", default=None, type=str,
        help="start from pretrained model"
    )

    parser.add_argument('data_path', type=str, help="path to ECGs' directory")
    parser.add_argument('stats_path', type=str)
    if args is None:
        parsed_args = parser.parse_args()
    else:
        parsed_args = parser.parse_args(args)

    # print(f'@{parsed_args.stats_path}/args.txt')
    """
    parsed_args = parser.parse_args(
        f'@./local-test-experiment/args.txt', namespace=parsed_args
    )
    """
    return parsed_args
