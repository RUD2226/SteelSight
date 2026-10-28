
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from data import create_dataloaders
from model import build_model


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
HISTORY_PATH = REPORT_DIR / "training_history.csv"
CURVE_PATH = REPORT_DIR / "training_curves.png"
METRICS_PATH = REPORT_DIR / "best_validation_metrics.json"

SEED = 42
EPOCHS = 12
PATIENCE = 4

LR_LAYER4 = 1e-4
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 3. Training and evaluation helpers
# --------------------------------------------------

def set_training_mode(model):
    """
    Train layer4 and the classification head while keeping
    early frozen layers and their BatchNorm statistics fixed.
    """
    model.train()

    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()


def train_one_epoch(model, loader, criterion, optimizer):
    set_training_mode(model)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images,
    )


@torch.inference_mode()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_images = 0

    all_targets = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        predictions = outputs.argmax(dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / total_images
    accuracy = accuracy_score(all_targets, all_predictions)

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# 4. Save training history and plot
# --------------------------------------------------

def save_history(history):
    if not history:
        return

    with HISTORY_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(history[0].keys())
        )
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        marker="o",
        label="Training loss",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        marker="o",
        label="Validation loss",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        marker="o",
        label="Training accuracy",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        marker="o",
        label="Validation accuracy",
    )
    axes[1].plot(
        epochs,
        [row["val_macro_f1"] for row in history],
        marker="o",
        label="Validation macro-F1",
    )
    axes[1].set_title("Classification performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=160)
    plt.close(fig)


# --------------------------------------------------
# 5. Main training routine
# --------------------------------------------------

def main():
    set_seed()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("STEELSIGHT: RESNET18 TRAINING")
    print("=" * 55)
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    # We do not use test_loader in this training routine.
    del test_loader

    # The first run may download pretrained ImageNet weights.
    model = build_model(
        num_classes=len(class_names),
        pretrained=True,
    ).to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nClasses: {class_names}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer4.parameters(),
                "lr": LR_LAYER4,
            },
            {
                "params": model.fc.parameters(),
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_val_f1 = -1.0
    best_metrics = {}
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss, val_accuracy, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step(val_f1)

        elapsed = time.perf_counter() - start_time

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "val_macro_f1": round(val_f1, 6),
            "lr_layer4": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "seconds": round(elapsed, 2),
        }

        history.append(row)
        save_history(history)

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f} | "
            f"Val macro-F1: {val_f1:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            best_metrics = {
                "best_epoch": epoch,
                "validation_accuracy": float(val_accuracy),
                "validation_macro_f1": float(val_f1),
                "validation_loss": float(val_loss),
            }

            # Save weights and all preprocessing/class metadata
            # necessary to reconstruct the model for inference.
            torch.save(
                {
                    "architecture": "resnet18",
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": 224,
                    "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225],
                    "best_epoch": epoch,
                    "validation_accuracy": float(val_accuracy),
                    "validation_macro_f1": float(val_f1),
                },
                CHECKPOINT_PATH,
            )

            with METRICS_PATH.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(best_metrics, file, indent=4)

            print("  -> New best checkpoint saved.")

        else:
            epochs_without_improvement += 1
            print(
                f"  -> No macro-F1 improvement "
                f"({epochs_without_improvement}/{PATIENCE})."
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation macro-F1 has plateaued.")
            break

    save_history(history)

    print("\n" + "=" * 55)
    print("TRAINING FINISHED")
    print("=" * 55)
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Best metrics: {best_metrics}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print(f"Training curves: {CURVE_PATH}")


if __name__ == "__main__":
    main()
