
from collections import Counter
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms




PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGES = (
    PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "train" / "images"
)

TEST_IMAGES = (
    PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "validation" / "images"
)

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.15

# Use zero workers initially for reliable Windows execution.
NUM_WORKERS = 0




IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms():
    """
    Training images receive augmentation.
    Internal-validation and test images do not.
    """

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return train_transform, eval_transform




def print_class_counts(split_name, labels, class_names):
    """Print the number of images in each class."""

    counts = Counter(labels)

    print(f"\n{split_name}: {len(labels)} images")

    for class_index, class_name in enumerate(class_names):
        print(f"  {class_name:18s}: {counts[class_index]}")




def create_dataloaders(batch_size=BATCH_SIZE):
    """
    Returns:
        train_loader: augmented training images
        val_loader: internal validation images
        test_loader: untouched provided validation images
        class_names: defect labels in model-output order
    """

    # Validate dataset directories.
    if not TRAIN_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Training image directory not found: {TRAIN_IMAGES}"
        )

    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Test image directory not found: {TEST_IMAGES}"
        )

    train_transform, eval_transform = get_transforms()

    # Same training images, with different transforms for train/validation.
    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_IMAGES),
        transform=train_transform,
    )

    internal_val_dataset = datasets.ImageFolder(
        root=str(TRAIN_IMAGES),
        transform=eval_transform,
    )

    # The provided validation directory is reserved for final testing.
    test_dataset = datasets.ImageFolder(
        root=str(TEST_IMAGES),
        transform=eval_transform,
    )

    class_names = train_dataset.classes

    # Ensure every split uses the same label-to-index mapping.
    if train_dataset.class_to_idx != internal_val_dataset.class_to_idx:
        raise RuntimeError("Training and validation class mappings differ.")

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise RuntimeError("Training and test class mappings differ.")

    if len(class_names) != 6:
        raise ValueError(
            f"Expected 6 defect classes, found {len(class_names)}: "
            f"{class_names}"
        )

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("One or more image datasets are empty.")

    # Reproducible, stratified split of the original training images.
    all_indices = list(range(len(train_dataset)))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=train_dataset.targets,
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(internal_val_dataset, val_indices)

    # Deterministic shuffle order across runs.
    generator = torch.Generator().manual_seed(SEED)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
    )

    # Print and verify the split distributions.
    train_labels = [train_dataset.targets[i] for i in train_indices]
    val_labels = [
        internal_val_dataset.targets[i] for i in val_indices
    ]
    test_labels = test_dataset.targets

    print("\n========== STEELSIGHT DATASET ==========")
    print(f"Dataset root: {PROJECT_ROOT}")
    print(f"Classes: {class_names}")

    print_class_counts("Training split", train_labels, class_names)
    print_class_counts("Internal validation split", val_labels, class_names)
    print_class_counts("Held-out test split", test_labels, class_names)

    print("\n========== DATALOADERS ==========")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {batch_size}")

    return train_loader, val_loader, test_loader, class_names




if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = (
        create_dataloaders()
    )

    images, labels = next(iter(train_loader))

    print("\n========== BATCH TEST ==========")
    print(f"Image tensor shape: {tuple(images.shape)}")
    print(f"Label tensor shape: {tuple(labels.shape)}")
    print(f"Label indices in batch: {labels[:10].tolist()}")
    print(f"Class names: {class_names}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    assert images.ndim == 4, "Expected a 4D image batch."
    assert images.shape[1] == 3, "Expected 3 image channels."
    assert images.shape[2:] == (IMAGE_SIZE, IMAGE_SIZE)
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]

    print("\nSUCCESS: Dataset loading and batch checks passed.")
