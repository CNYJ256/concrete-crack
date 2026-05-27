"""Model factory for concrete crack detection."""

from src.models.unet import Unet
from src.models.fcn8s import FCN8s
from src.models.deeplabv3plus import DeepLabV3Plus


def create_model(name: str, num_classes: int = 1):
    """Create a segmentation model by name.

    Args:
        name: Model identifier ("unet", "fcn8s", "deeplabv3plus").
        num_classes: Number of output channels (default 1 for raw logits).

    Returns:
        nn.Module instance.

    Raises:
        ValueError: If model name is unknown.
    """
    if name == "unet":
        return Unet(3, num_classes)
    elif name == "deeplabv3plus":
        return DeepLabV3Plus(num_classes)
    elif name == "fcn8s":
        return FCN8s(num_classes)
    else:
        raise ValueError(
            f"Unknown model: {name}. Expected one of: unet, deeplabv3plus, fcn8s"
        )
