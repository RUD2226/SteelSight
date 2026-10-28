
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import create_dataloaders
from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "gradcam_examples"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224


class ActivationGradientHook:
    """Capture feature maps and their gradients from a target layer."""

    def __init__(self, target_layer):
        self.activations = None
        self.gradients = None

        self.handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        self.gradients = gradient

    def close(self):
        self.handle.remove()


def compute_gradcam(model, image_tensor, target_class, hook):
    """Compute a Grad-CAM heatmap for one selected class."""

    model.zero_grad(set_to_none=True)

    # Gradients must be enabled for Grad-CAM.
    with torch.enable_grad():
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        predicted_class = int(logits.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_class].item())

        # Differentiate the selected predicted-class score.
        score = logits[0, target_class]
        score.backward()

        activations = hook.activations
        gradients = hook.gradients

        if activations is None or gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients."
            )

        # Average gradients across the feature-map spatial dimensions.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of feature maps, followed by ReLU.
        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        # Resize the heatmap to the input image dimensions.
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()

        # Normalize for visualization.
        cam -= cam.min()
        cam_max = cam.max()

        if cam_max > 1e-8:
            cam /= cam_max

    return cam, predicted_class, confidence


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recreate the held-out data loader and original class ordering.
    _, _, test_loader, class_names = create_dataloaders()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["class_names"] != class_names:
        raise ValueError(
            "Checkpoint labels do not match the current dataset."
        )

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # First identify correct and incorrect predictions.
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            logits = model(images)

            all_predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    test_dataset = test_loader.dataset
    test_targets = list(test_dataset.targets)

    if len(all_predictions) != len(test_dataset):
        raise RuntimeError("Prediction count differs from dataset size.")

    # Select one correctly classified example per class.
    representative = {}

    # Include every misclassified image to explain the failure cases.
    mistakes = []

    for index, (true_label, prediction) in enumerate(
        zip(test_targets, all_predictions)
    ):
        if true_label != prediction:
            mistakes.append(index)
        elif true_label not in representative:
            representative[true_label] = index

    selected_indices = [
        representative[label]
        for label in range(len(class_names))
        if label in representative
    ]

    selected_indices.extend(mistakes)

    if not selected_indices:
        raise RuntimeError("No examples were selected for visualization.")

    # The final convolutional block is an appropriate Grad-CAM target.
    hook = ActivationGradientHook(model.layer4[-1])

    results = []

    try:
        for index in selected_indices:
            image_tensor, true_label = test_dataset[index]
            image_path = test_dataset.samples[index][0]

            # Explain the class actually predicted by the model.
            predicted_label = all_predictions[index]

            cam, predicted_class, confidence = compute_gradcam(
                model,
                image_tensor,
                predicted_label,
                hook,
            )

            # Load an unnormalized image for display.
            original = (
                Image.open(image_path)
                .convert("RGB")
                .resize((IMAGE_SIZE, IMAGE_SIZE))
            )
            original = np.asarray(original)

            correct = true_label == predicted_class

            results.append({
                "image": original,
                "cam": cam,
                "true_label": class_names[true_label],
                "predicted_label": class_names[predicted_class],
                "confidence": confidence,
                "correct": correct,
                "index": index,
            })

            print(
                f"Image {index}: "
                f"True={class_names[true_label]} | "
                f"Predicted={class_names[predicted_class]} | "
                f"Confidence={confidence:.3f} | "
                f"Correct={correct}"
            )

    finally:
        hook.close()

    # Create one grid containing representative cases and mistakes.
    columns = 2
    rows = math.ceil(len(results) / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 4.5 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, result in zip(axes.flat, results):
        ax.imshow(result["image"])
        ax.imshow(
            result["cam"],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        status = "Correct" if result["correct"] else "MISCLASSIFIED"

        ax.set_title(
            f"True: {result['true_label']}\n"
            f"Predicted: {result['predicted_label']} "
            f"({result['confidence']:.1%}) | {status}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        "SteelSight — Grad-CAM Visual Explanations",
        fontsize=15,
    )
    fig.tight_layout()

    grid_path = OUTPUT_DIR / "gradcam_grid.png"
    fig.savefig(grid_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save each visualization separately as well.
    for position, result in enumerate(results, start=1):
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.imshow(result["image"])
        ax.imshow(
            result["cam"],
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=1,
        )

        status = "Correct" if result["correct"] else "MISCLASSIFIED"

        ax.set_title(
            f"True: {result['true_label']} | "
            f"Predicted: {result['predicted_label']} "
            f"({result['confidence']:.1%})\n{status}"
        )
        ax.axis("off")

        filename = (
            f"example_{position:02d}_"
            f"{result['true_label']}_"
            f"pred_{result['predicted_label']}.png"
        )

        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    print("\nGrad-CAM generation complete.")
    print(f"Examples generated: {len(results)}")
    print(f"Combined grid: {grid_path}")
    print(f"Individual images: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
