#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepweeds_pytorch.data import build_loaders, class_weights_from_csv
from deepweeds_pytorch.engine import evaluate_epoch, save_history, set_seed, train_one_epoch
from deepweeds_pytorch.model import build_resnet18
from deepweeds_pytorch.reporting import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on DeepWeeds.")
    parser.add_argument("--data-root", default="deepweeds_pytorch/data")
    parser.add_argument("--output-dir", default="deepweeds_pytorch/results/run_01")
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weighted", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(
        args.data_root,
        fold=args.fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    num_classes = len(loaders.class_names)

    model = build_resnet18(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    weight = None
    if args.class_weighted:
        train_csv = Path(args.data_root) / "labels" / f"train_subset{args.fold}.csv"
        weight = class_weights_from_csv(train_csv, num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    bad_epochs = 0
    history = []
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, loaders.train, criterion, optimizer, device, scaler
        )
        val_stats = evaluate_epoch(model, loaders.val, criterion, device)
        scheduler.step(val_stats.macro_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_stats.loss,
            "train_accuracy": train_stats.accuracy,
            "train_macro_f1": train_stats.macro_f1,
            "val_loss": val_stats.loss,
            "val_accuracy": val_stats.accuracy,
            "val_macro_f1": val_stats.macro_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_stats.loss:.4f} "
            f"val_loss={val_stats.loss:.4f} "
            f"val_acc={val_stats.accuracy:.4f} "
            f"val_macro_f1={val_stats.macro_f1:.4f}"
        )

        if val_stats.macro_f1 > best_f1:
            best_f1 = val_stats.macro_f1
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": loaders.class_names,
                    "fold": args.fold,
                    "image_size": args.image_size,
                    "best_val_macro_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("early stopping")
                break

    history_path = output_dir / "history.csv"
    save_history(history, history_path)
    plot_training_history(history_path, output_dir / "training_curves.png")
    print(f"best validation macro F1: {best_f1:.4f}")
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
