
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes=6, pretrained=True):
    """
    Build a ResNet18 classifier for industrial surface defects.

    Frozen:
        Early ResNet feature-extraction layers.

    Trainable:
        Layer 4 and the new classification head.
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Freeze the pretrained parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final convolutional block.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Replace ImageNet's 1000-class head with our six defect classes.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # The replacement classification head must be trainable.
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model
