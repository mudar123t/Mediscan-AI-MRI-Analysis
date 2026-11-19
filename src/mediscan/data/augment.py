"""Training augmentations (albumentations). Geometric transforms move image and mask together."""

from __future__ import annotations

import albumentations as A
import cv2


def train_transform(strength: str = "default") -> A.Compose:
    """Augmentation pipeline for training.

    Args:
        strength: ``"default"`` (semester-2 recipe) or ``"none"`` (old recipe,
            used for the fair semester-1 baseline).
    """
    if strength == "none":
        return A.Compose([])
    if strength != "default":
        raise ValueError(f"unknown augmentation strength: {strength}")
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.7,
            ),
            A.ElasticTransform(alpha=20, sigma=5, border_mode=cv2.BORDER_CONSTANT, p=0.15),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ]
    )
