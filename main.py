"""Training and evaluation entry point for VGP-Net.

The training harness is adapted from the learned correspondence release.
"""
# Code is heavily borrowed from https://github.com/vcg-uvic/learned-correspondence-release
# Author: Jiahui Zhang
# Date: 2019/09/03
# E-mail: jiahui-z15@mails.tsinghua.edu.cn

import os
import sys
from multiprocessing import freeze_support
from pathlib import Path

from config import get_config, print_usage


def create_log_dir(config):
    """Create the existing train/valid/test layout and save its configuration."""
    import torch

    result_path = Path(config.log_base)
    for name in ("train", "valid", "test"):
        (result_path / name).mkdir(parents=True, exist_ok=True)

    config.log_path = str(result_path / "train")
    config_path = result_path / "config.th"
    if config_path.exists():
        print(f"Warning: overwriting configuration at {config_path}")
    torch.save(config, config_path)


def main(config):
    """Run VGP-Net with the existing dataset, loss, train, and test interfaces."""
    if config.run_mode not in ("train", "test"):
        raise ValueError("run_mode must be 'train' or 'test'.")

    data_options = ("data_tr", "data_va") if config.run_mode == "train" else ("data_te",)
    for option in data_options:
        data_path = Path(getattr(config, option))
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}. Set --{option} to an existing HDF5 file."
            )

    # Set GPU visibility before importing PyTorch or the model.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    import torch
    from torch.utils.data import DataLoader
    from VGPNet import VGPNet as Model

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The current training/test harness requires CUDA. "
            "Use a CUDA-enabled PyTorch installation and an available GPU."
        )

    from data import CorrespondencesDataset, collate_fn

    model = Model(config).cuda()
    print(f"VGP-Net: {config.run_mode} (CUDA_VISIBLE_DEVICES={config.gpu_id})")

    if config.run_mode == "train":
        from train import train

        create_log_dir(config)
        train_dataset = CorrespondencesDataset(config.data_tr, config)
        train_loader = DataLoader(
            train_dataset, batch_size=config.train_batch_size, shuffle=True,
            num_workers=16, pin_memory=False, collate_fn=collate_fn,
        )

        valid_dataset = CorrespondencesDataset(config.data_va, config)
        valid_loader = DataLoader(
            valid_dataset, batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn,
        )
        print("Starting training...")
        train(model, train_loader, valid_loader, config)
    else:
        from test import test

        test_dataset = CorrespondencesDataset(config.data_te, config)
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=8, pin_memory=False, collate_fn=collate_fn,
        )
        test(test_loader, model, config)


if __name__ == "__main__":
    freeze_support()
    config, unparsed = get_config()
    if unparsed:
        print_usage()
        print(f"Unrecognized arguments: {' '.join(unparsed)}", file=sys.stderr)
        sys.exit(1)
    main(config)
