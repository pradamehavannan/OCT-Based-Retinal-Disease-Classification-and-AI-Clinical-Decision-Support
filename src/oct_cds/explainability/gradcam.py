"""Grad-CAM overlay for the CDS report. STUB — fill in with pytorch-grad-cam
(installed via the ``explain`` extra) once a checkpoint exists.
"""

from __future__ import annotations

import numpy as np


def default_target_layer(model):
    """Best-effort last conv layer for the supported backbones."""
    name = type(model).__name__.lower()
    if "densenet" in name:
        return model.features.norm5
    if "resnet" in name:
        return model.layer4[-1]
    if "efficientnet" in name:
        return model.conv_head
    # timm models: fall back to the last module with weight of ndim 4
    last = None
    for m in model.modules():
        if hasattr(m, "weight") and getattr(m.weight, "ndim", 0) == 4:
            last = m
    return last


def gradcam_heatmap(model, image_tensor, target_class: int | None = None) -> np.ndarray:
    """Return a HxW heatmap in [0, 1]. Raises if the ``explain`` extra is missing."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install extras: pip install '.[explain]'") from exc

    import torch

    cam = GradCAM(model=model, target_layers=[default_target_layer(model)])
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    batch = image_tensor.unsqueeze(0) if image_tensor.ndim == 3 else image_tensor
    grayscale = cam(input_tensor=batch, targets=targets)[0]
    return np.asarray(grayscale, dtype=np.float32)
