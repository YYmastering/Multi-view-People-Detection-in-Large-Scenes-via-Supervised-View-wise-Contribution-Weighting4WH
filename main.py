import argparse
import os

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main(args):
    args.project_root_path = os.path.abspath(os.path.dirname(__file__))
    args.proj_root = args.project_root_path
    args.devices = [args.device, args.device]  # compatibility with other runners

    if args.world_reduce is None:
        args.world_reduce = 4 if args.dataset in {
            "warehouse", "wildtrack", "multiviewx"
        } else 2
    if args.img_reduce is None:
        args.img_reduce = 4 if args.dataset in {
            "warehouse", "wildtrack", "multiviewx"
        } else 2
    if args.map_sigma is None:
        args.map_sigma = 5.0 / args.world_reduce \
            if args.dataset == "warehouse" else 3.0
    if args.fix_2D is None:
        args.fix_2D = int(args.variant != "2D")
    if args.fix_svp is None:
        args.fix_svp = int(args.variant == "2D_SVP_VCW")

    default_thresholds = {
        "warehouse": (5.0, 10.0),
        "wildtrack": (20.0, 20.0),
        "multiviewx": (20.0, 20.0),
    }
    nms_default, distance_default = default_thresholds.get(
        args.dataset, (40.0, 80.0)
    )
    if args.nms_thres is None:
        args.nms_thres = nms_default
    if args.dist_thres is None:
        args.dist_thres = distance_default

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")

    if args.dataset in {"warehouse", "wildtrack", "multiviewx"}:
        from x_training.W_M.model_run import model_run
    elif args.dataset == "citystreet":
        from x_training.CityStreet.model_run import model_run
    elif args.dataset == "cvcs":
        from x_training.CVCS.model_run import model_run
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")
    model_run(args)


def build_parser():
    parser = argparse.ArgumentParser(description="Multiview VCW detector")
    parser.add_argument(
        "--variant", default="2D_SVP_VCW",
        choices=["2D", "2D_SVP", "2D_SVP_VCW"],
        help="Training stage: image, supervised view prediction, or full VCW.",
    )
    parser.add_argument(
        "--arch", default="resnet18", choices=["vgg16", "resnet18"]
    )
    parser.add_argument(
        "-d", "--dataset", default="warehouse",
        choices=["warehouse", "wildtrack", "multiviewx", "citystreet", "cvcs"],
    )
    parser.add_argument(
        "--data_path", default="/home/student/YY_new/rfdetr_modify/warehouse_08",
        help="Warehouse root containing split directories.",
    )
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--test_split", default="test_valid_merged")
    parser.add_argument("--data_root", default="/mnt/data/Datasets")
    parser.add_argument("--pretrain", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--test", default=None, metavar="CHECKPOINT",
        help="Skip training and evaluate this checkpoint.",
    )
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_pretrained_backbone", action="store_true")

    parser.add_argument("-j", "--num_workers", type=int, default=4)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--val_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay", "--ld", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler", "--lrs", default="onecycle",
        choices=["onecycle", "lambda", "step"],
    )
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")

    parser.add_argument("--facofmaxgt", "--fm", type=float, default=100)
    parser.add_argument("--fix_2D", type=int, choices=[0, 1], default=None)
    parser.add_argument("--fix_svp", type=int, choices=[0, 1], default=None)
    parser.add_argument("--fix_weight", type=int, choices=[0, 1], default=0)
    parser.add_argument("--weight_svp", type=float, default=1)
    parser.add_argument("--weight_2D", type=float, default=1)
    parser.add_argument("--img_reduce", "--ir", type=int, default=None)
    parser.add_argument("--world_reduce", "--wr", type=int, default=None)
    parser.add_argument("--map_sigma", "--ms", type=float, default=None)
    parser.add_argument("--facofmaxgt_gp", "--fmg", type=float, default=10)
    parser.add_argument("--lrfac", type=float, default=0.01)

    parser.add_argument("--cls_thres", "--ct", type=float, default=0.4)
    parser.add_argument("--nms_thres", "--nt", type=float, default=None)
    parser.add_argument("--dist_thres", "--dt", type=float, default=None)
    parser.add_argument("--person_heights", "--ph", nargs="+", type=float,
                        default=[1750])
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
