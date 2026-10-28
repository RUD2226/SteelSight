
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data import create_dataloaders
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
REPORT_DIR = PROJECT_ROOT / "reports"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@torch.inference_mode()
def evaluate_model(model, loader, class_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_targets = []
    all_predictions = []
    total_loss = 0.0
    total_images = 0

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    labels_range = list(range(len(class_names)))

    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        labels=labels_range,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_targets,
        all_predictions,
        labels=labels_range,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=labels_range,
    )

    metrics = {
        "model": "ResNet18",
        "device": str(DEVICE),
        "test_images": total_images,
        "test_loss": float(total_loss / total_images),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "evaluation_seconds": round(elapsed, 3),
        "class_names": class_names,
    }

    return metrics, report, matrix


def save_confusion_matrix(matrix, class_names):
    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="SteelSight — Test Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "test_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run src\\train.py first."
        )

    print(f"Evaluation device: {DEVICE}")

    # Reconstruct the test loader and label mapping.
    _, _, test_loader, class_names = create_dataloaders()

    # This checkpoint only contains tensors and basic Python values.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["class_names"] != class_names:
        raise ValueError(
            "Checkpoint class order differs from dataset class order."
        )

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nEvaluating the held-out test set...")

    metrics, report, matrix = evaluate_model(
        model,
        test_loader,
        class_names,
    )

    # Save numerical metrics.
    with (REPORT_DIR / "test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    with (REPORT_DIR / "test_classification_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Save a convenient CSV for the resume and README.
    fields = ["class", "precision", "recall", "f1-score", "support"]

    with (REPORT_DIR / "test_classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for name in class_names:
            writer.writerow({"class": name, **report[name]})

        for name in ["macro avg", "weighted avg"]:
            writer.writerow({"class": name, **report[name]})

    save_confusion_matrix(matrix, class_names)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Test images: {metrics['test_images']}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['test_macro_f1']:.4f}")
    print(f"Evaluation time: {metrics['evaluation_seconds']:.2f}s")

    print("\n========== PER-CLASS REPORT ==========")
    print(
        classification_report(
            list(range(len(class_names))),
            [],  # The detailed report is printed from saved test results below.
        )
    ) if False else None

    for name in class_names:
        row = report[name]
        print(
            f"{name:18s} | "
            f"Precision: {row['precision']:.3f} | "
            f"Recall: {row['recall']:.3f} | "
            f"F1: {row['f1-score']:.3f} | "
            f"Support: {int(row['support'])}"
        )

    print("\n========== CONFUSION MATRIX ==========")
    print(matrix)

    print("\nSaved reports:")
    print(REPORT_DIR / "test_metrics.json")
    print(REPORT_DIR / "test_classification_report.csv")
    print(REPORT_DIR / "test_confusion_matrix.png")


if __name__ == "__main__":
    main()
