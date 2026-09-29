"""Train ResEmoteNet on RAF-DB arranged as an ImageFolder dataset.

Expected directory layout:

    /mnt/data/yanyi2025/cyj/raf-db/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

The script uses the validation split for model selection because the shown
dataset has no separate test directory. GPU selection happens before importing
torch, so ``--gpu 0`` refers to physical GPU 0.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data/yanyi2025/cyj/raf-db"),
        help="RAF-DB root containing train and val directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/data/yanyi2025/cyj/raf-db/resemotenet"),
        help="Directory for the best checkpoint and metrics.",
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="Physical GPU id to use. Use 'cpu' to disable CUDA.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="Validation epochs without improvement before LR is multiplied by 0.1.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Use 0 to disable early stopping; otherwise stop after this many non-improving epochs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


args = parse_args()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if args.gpu.lower() == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from approach.ResEmoteNet import ResEmoteNet


IMAGE_SIZE = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, class_names):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
        },
        path,
    )


def main():
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        split for split in ("train", "val")
        if not (data_root / split).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing RAF-DB directories under {data_root}: {', '.join(missing)}"
        )

    if args.gpu.lower() == "cpu":
        device = torch.device("cpu")
    elif not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and PyTorch installation."
        )
    else:
        device = torch.device("cuda:0")

    train_transform, eval_transform = make_transforms()
    train_dataset = datasets.ImageFolder(
        data_root / "train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        data_root / "val", transform=eval_transform
    )

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Class folders differ between train and val: "
            f"train={train_dataset.class_to_idx}, val={val_dataset.class_to_idx}"
        )

    class_names = train_dataset.classes
    if len(class_names) != 7:
        raise ValueError(
            f"Expected 7 emotion classes, found {len(class_names)}: {class_names}"
        )

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, pin_memory
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, pin_memory
    )

    model = ResEmoteNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=args.scheduler_patience,
    )

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(
            f"CUDA device: {torch.cuda.get_device_name(0)} "
            f"(physical GPU {args.gpu})"
        )
    print(f"Dataset: {data_root}")
    print(f"Classes: {json.dumps(class_names, ensure_ascii=True)}")
    print(f"Samples: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = []
    best_val_acc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                epoch,
                best_val_acc,
                class_names,
            )
            print(f"  Saved best checkpoint (val acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if (
                args.early_stopping_patience > 0
                and patience_counter >= args.early_stopping_patience
            ):
                print("Early stopping: validation accuracy did not improve.")
                break

    checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    best_val_loss, best_val_acc = run_epoch(
        model, val_loader, criterion, device
    )

    val_metrics = {
        "checkpoint_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_acc"],
        "best_checkpoint_val_loss": best_val_loss,
        "best_checkpoint_val_accuracy": best_val_acc,
        "class_names": class_names,
    }
    (output_dir / "val_metrics.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"Best checkpoint epoch {checkpoint['epoch']} | "
        f"val acc {best_val_acc:.4f}"
    )
    print(f"Saved metrics to {output_dir / 'metrics.csv'}")
    print(f"Saved checkpoint to {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
