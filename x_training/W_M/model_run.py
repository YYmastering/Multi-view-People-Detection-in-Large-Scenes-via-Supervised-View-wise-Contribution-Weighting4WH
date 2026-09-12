import datetime
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms

from datasets.W_M.MultiviewX import MultiviewX
from datasets.W_M.Warehouse import WarehouseCocoSplit
from datasets.W_M.Wildtrack import Wildtrack
from datasets.W_M.frameDataset_head import frameDataset as WMFrameDataset
from datasets.W_M.frameDataset_warehouse import frameDataset as WarehouseFrameDataset
from models.W_M.WM_Detector import PerspTransDetector
from trainer.W_M.WM_trainer import PerspectiveTrainer
from utils.logger import Logger


def _load_model_weights(model, path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        print(f"Missing checkpoint keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return checkpoint


def _make_datasets(args, transform):
    if args.dataset == "warehouse":
        train_base = WarehouseCocoSplit(args.data_path, args.train_split)
        val_base = WarehouseCocoSplit(args.data_path, args.test_split)
        common = dict(
            _transform=transform,
            grid_reduce=args.world_reduce,
            img_reduce=args.img_reduce,
            facofmaxgt=args.facofmaxgt,
            map_sigma=args.map_sigma,
        )
        return (
            WarehouseFrameDataset(train_base, **common),
            WarehouseFrameDataset(val_base, **common),
        )

    if args.dataset == "wildtrack":
        base = Wildtrack(os.path.join(args.data_root, "Wildtrack"))
    elif args.dataset == "multiviewx":
        base = MultiviewX(os.path.join(args.data_root, "MultiviewX"))
    else:
        raise ValueError("W/M runner supports warehouse, wildtrack, multiviewx")
    return (
        WMFrameDataset(
            base, train=True, _transform=transform,
            grid_reduce=args.world_reduce, img_reduce=args.img_reduce,
            facofmaxgt=args.facofmaxgt,
        ),
        WMFrameDataset(
            base, train=False, _transform=transform,
            grid_reduce=args.world_reduce, img_reduce=args.img_reduce,
            facofmaxgt=args.facofmaxgt,
        ),
    )


def _make_scheduler(args, optimizer, train_loader):
    if args.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
        )
    if args.lr_scheduler == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch),
        )
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    raise ValueError("lr_scheduler must be onecycle, lambda, or step")


def model_run(args):
    if args.variant != "2D" and not args.pretrain and not args.resume and not args.test:
        raise ValueError(
            f"{args.variant} is a staged model: supply the preceding stage "
            "with --pretrain, or continue with --resume"
        )
    if args.pretrain and args.resume:
        raise ValueError("use either --pretrain or --resume, not both")
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    normalize = transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )
    image_transform = transforms.Compose([
        transforms.Resize([720, 1280]),
        transforms.ToTensor(),
        normalize,
    ])
    train_set, val_set = _make_datasets(args, image_transform)

    pin_memory = str(args.device).startswith("cuda")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or timestamp
    if args.resume and args.run_name is None:
        logdir = os.path.dirname(os.path.abspath(args.resume))
    else:
        logdir = os.path.join(
            args.project_root_path, "logs", f"{args.dataset}_dataset",
            args.arch, args.variant, run_name,
        )
    os.makedirs(logdir, exist_ok=True)
    sys.stdout = Logger(os.path.join(logdir, "log.txt"))
    print("Settings:")
    print(vars(args))
    print(f"Train frames: {len(train_set)}, validation frames: {len(val_set)}")
    print(f"Outputs: {logdir}")

    model = PerspTransDetector(train_set, args)
    checkpoint_to_test = args.test
    if checkpoint_to_test:
        print(f"Loading test checkpoint: {checkpoint_to_test}")
        _load_model_weights(model, checkpoint_to_test, args.device)
        trainer = PerspectiveTrainer(model, args, logdir)
        trainer.validate(
            val_loader, os.path.join(logdir, "test.txt"), epoch="test"
        )
        return

    if args.pretrain:
        print(f"Loading preceding-stage weights: {args.pretrain}")
        _load_model_weights(model, args.pretrain, args.device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable parameters remain after freezing")
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = _make_scheduler(args, optimizer, train_loader)
    start_epoch = 1
    if args.resume:
        print(f"Resuming checkpoint: {args.resume}")
        checkpoint = _load_model_weights(model, args.resume, args.device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    trainer = PerspectiveTrainer(model, args, logdir)
    latest_path = os.path.join(
        logdir, f"latest_{args.variant}_model.pth"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        trainer.train(train_loader, epoch, optimizer, scheduler)
        trainer.save_checkpoint(latest_path, epoch, optimizer, scheduler)

        should_validate = epoch % args.val_epochs == 0 or epoch == args.epochs
        if not should_validate:
            continue
        result_path = (
            os.path.join(logdir, f"test_epoch_{epoch:03d}.txt")
            if args.variant == "2D_SVP_VCW" else None
        )
        metrics = trainer.validate(
            val_loader, result_path=result_path, epoch=epoch
        )
        trainer.save_checkpoint(
            latest_path, epoch, optimizer, scheduler, metrics
        )
        if args.variant == "2D_SVP_VCW" and metrics["moda"] > trainer.best_moda:
            trainer.best_moda = metrics["moda"]
            trainer.save_checkpoint(
                os.path.join(logdir, "best_model.pth"),
                epoch, optimizer, scheduler, metrics,
            )
