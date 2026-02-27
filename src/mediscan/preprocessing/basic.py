from __future__ import annotations
import numpy as np
import cv2

def normalize_01_uint8(img_u8: np.ndarray) -> np.ndarray:
    """
    Input: uint8 image [0..255]
    Output: float32 image [0..1]
    """
    if img_u8.dtype != np.uint8:
        img_u8 = img_u8.astype(np.uint8)
    return (img_u8.astype(np.float32) / 255.0)

def binarize_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    Your masks have many values (0,1,2,3,...).
    For pilot: tumor = any value > 0
    Output: uint8 mask with values {0,255}
    """
    tumor = (mask_u8 > 0).astype(np.uint8)
    return tumor * 255

def gaussian_denoise(img01: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Optional denoise on normalized image [0..1].
    """
    if ksize <= 1:
        return img01
    # OpenCV expects float32 is ok
    return cv2.GaussianBlur(img01, (ksize, ksize), 0)

def to_uint8(img01: np.ndarray) -> np.ndarray:
    """
    Convert float [0..1] -> uint8 [0..255] for saving.
    """
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0).round().astype(np.uint8)
