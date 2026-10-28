
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import build_model



PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.set_page_config(
    page_title="SteelSight | Industrial AI",
    page_icon="🔬",
    layout="wide",
)


def friendly_name(name):
    return name.replace("_", " ").replace("-", " ").title()



@st.cache_resource(show_spinner="Loading SteelSight model...")
def load_artifacts(checkpoint_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_size = checkpoint.get("image_size", 224)

    mean = checkpoint.get(
        "normalization_mean",
        [0.485, 0.456, 0.406],
    )

    std = checkpoint.get(
        "normalization_std",
        [0.229, 0.224, 0.225],
    )

    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, class_names, device, preprocessing, image_size




@torch.inference_mode()
def predict(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    logits = model(image_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_seconds = time.perf_counter() - start

    probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
    predicted_index = int(np.argmax(probabilities))

    return probabilities, predicted_index, inference_seconds




def compute_gradcam(model, image_tensor, target_index, device, image_size):
    activations = {}
    gradients = {}

    def save_gradient(gradient):
        gradients["value"] = gradient

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(save_gradient)

    handle = model.layer4[-1].register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        input_batch = image_tensor.unsqueeze(0).to(device)

        with torch.enable_grad():
            logits = model(input_batch)
            selected_score = logits[0, target_index]
            selected_score.backward()

            feature_maps = activations["value"]
            feature_gradients = gradients["value"]

            weights = feature_gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            cam = torch.relu(
                (weights * feature_maps).sum(dim=1, keepdim=True)
            )

            cam = F.interpolate(
                cam,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam = cam.detach().cpu().numpy()

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    cam -= cam.min()
    maximum = cam.max()

    if maximum > 1e-8:
        cam /= maximum

    return cam


def make_overlay(image, cam, image_size):
    image_resized = image.convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    original_array = np.asarray(image_resized).astype(np.float32)

    heatmap_array = (
        plt.get_cmap("jet")(cam)[..., :3] * 255
    ).astype(np.float32)

    overlay = (
        0.55 * original_array + 0.45 * heatmap_array
    ).clip(0, 255).astype(np.uint8)

    return image_resized, overlay




st.title("SteelSight")
st.subheader("Explainable Industrial Steel Defect Classification")

st.markdown(
    "Upload a steel-surface image to classify its defect type, "
    "inspect class scores, and visualize the image regions "
    "influencing the prediction."
)

st.divider()

if not CHECKPOINT_PATH.exists():
    st.error(
        "Trained model checkpoint not found. "
        "Run `python src\\train.py` first."
    )
    st.stop()

try:
    model, class_names, device, preprocessing, image_size = (
        load_artifacts(str(CHECKPOINT_PATH))
    )
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# Sidebar: model and evaluation information
with st.sidebar:
    st.header("Model information")

    st.write("**Architecture:** ResNet18")
    st.write("**Task:** Six-class classification")
    st.write(f"**Device:** {device}")
    st.write(f"**Number of classes:** {len(class_names)}")

    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open(
                "r", encoding="utf-8"
            ) as file:
                metrics = json.load(file)

            st.metric(
                "Held-out test accuracy",
                f"{metrics['test_accuracy']:.2%}",
            )

            st.metric(
                "Held-out test macro-F1",
                f"{metrics['test_macro_f1']:.2%}",
            )
        except (OSError, ValueError, KeyError):
            st.caption("Saved test metrics are unavailable.")

    st.divider()
    st.caption(
        "Grad-CAM provides a qualitative explanation of model "
        "attention. Its heatmap is not a physical defect boundary."
    )


uploaded_file = st.file_uploader(
    "Upload a steel-surface image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    help="Supported formats: JPG, PNG, BMP, and TIFF.",
)


if uploaded_file is None:
    st.info(
        "Choose an image to begin. For the most meaningful demo, "
        "start with images similar to the NEU steel-defect dataset."
    )

else:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()

    image_tensor = preprocessing(original_image)

    probabilities, predicted_index, inference_seconds = predict(
        model,
        image_tensor,
        device,
    )

    predicted_class = class_names[predicted_index]
    predicted_confidence = float(probabilities[predicted_index])

    st.divider()

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.subheader("Uploaded image")
        st.image(
            original_image,
            caption="Input steel-surface image",
            use_container_width=True,
        )

    with right:
        st.subheader("Prediction")

        st.metric(
            "Predicted defect",
            friendly_name(predicted_class),
        )

        st.metric(
            "Top-class score",
            f"{predicted_confidence:.2%}",
        )

        st.caption(
            "The score is the model's softmax output; it is not "
            "necessarily a calibrated probability of correctness."
        )

        st.caption(
            f"Inference time: {inference_seconds * 1000:.1f} ms "
            f"on {device}"
        )

    # Display scores for the three most likely classes.
    st.subheader("Top predictions")

    top_indices = np.argsort(probabilities)[::-1][:3]
    score_columns = st.columns(3)

    for column, index in zip(score_columns, top_indices):
        with column:
            st.metric(
                friendly_name(class_names[index]),
                f"{probabilities[index]:.2%}",
            )
            st.progress(float(probabilities[index]))

    with st.expander("View all six class scores"):
        for index in np.argsort(probabilities)[::-1]:
            st.write(
                f"**{friendly_name(class_names[index])}:** "
                f"{probabilities[index]:.2%}"
            )
            st.progress(float(probabilities[index]))

    if predicted_confidence < 0.60:
        st.warning(
            "The top-class score is relatively low. Treat this "
            "prediction cautiously and review the image."
        )

    st.divider()
    st.subheader("Grad-CAM explainability")

    target_index = class_names.index(predicted_class)

    try:
        target_label = st.selectbox(
            "Class to explain",
            options=class_names,
            index=target_index,
            format_func=friendly_name,
            help=(
                "By default, Grad-CAM explains the predicted class. "
                "You can select another class to inspect the regions "
                "that influence its score."
            ),
        )

        cam = compute_gradcam(
            model=model,
            image_tensor=image_tensor,
            target_index=class_names.index(target_label),
            device=device,
            image_size=image_size,
        )

        display_image, overlay = make_overlay(
            original_image,
            cam,
            image_size,
        )

        image_column, heatmap_column = st.columns(2, gap="large")

        with image_column:
            st.image(
                display_image,
                caption="Resized model input",
                use_container_width=True,
            )

        with heatmap_column:
            st.image(
                overlay,
                caption=f"Grad-CAM for {friendly_name(target_label)}",
                use_container_width=True,
            )

        st.caption(
            "Warm-colored regions indicate stronger positive "
            "contributions to the selected class score within this "
            "Grad-CAM visualization. Interpret these regions "
            "qualitatively, not as a guaranteed defect localization."
        )

    except Exception as exc:
        st.error(f"Could not generate Grad-CAM: {exc}")


st.divider()

st.caption(
    "SteelSight is an educational computer-vision prototype trained "
    "on NEU steel-surface images. Performance on real factory imagery, "
    "different cameras, materials, and lighting conditions may differ. "
    "Do not use this prototype as the sole basis for safety-critical "
    "quality-control decisions."
)
