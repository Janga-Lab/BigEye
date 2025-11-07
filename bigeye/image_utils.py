"""
Image Processing Utilities for DeepLab Training.
"""

import cv2
import numpy as np
from typing import Callable, Tuple

def _get_resizing_interpolation(original_shape: tuple, resize_shape: tuple):
    """Determines the best CV2 interpolation method."""
    original_size = np.prod(original_shape)
    resize_size = np.prod(resize_shape)

    if original_size > resize_size:
        return cv2.INTER_AREA  # Shrinking
    elif original_size < resize_size:
        return cv2.INTER_CUBIC # Expanding
    else:
        return cv2.INTER_LINEAR # Identical

def read_image_or_mask(file_name: str, as_grayscale=False, resize_dims=None) -> np.ndarray:
    """Reads and optionally resizes an image or mask."""
    file_name = str(file_name)
    use_resize_dims = (
        isinstance(resize_dims, tuple) and len(resize_dims) == 2
    )

    if as_grayscale:
        output = cv2.imread(file_name, 0)
        if output is None:
            raise IOError(f"Could not read grayscale image: {file_name}")
        if use_resize_dims:
            interpolation = cv2.INTER_NEAREST # Use NEAREST for masks
            output = cv2.resize(output, resize_dims, interpolation=interpolation)
    else:
        output = cv2.imread(file_name)
        if output is None:
            raise IOError(f"Could not read image: {file_name}")
        if use_resize_dims:
            interpolation = _get_resizing_interpolation(output.shape[:2], resize_dims)
            output = cv2.resize(output, resize_dims, interpolation=interpolation)
    
    return output

def _channelwise(func: Callable, image: np.ndarray) -> np.ndarray:
    """Applies a function to each channel of an image."""
    return cv2.merge([func(channel) for channel in cv2.split(image)])

def apply_clahe(image: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Converts to LAB color space, applies CLAHE to the L-channel,
    and converts back to BGR.
    """
    def _clahe(image_channel_data: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image_channel_data)
    
    # Check if image is grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        if len(image.shape) == 3:
             # Squeeze single channel to 2D
             image = image.squeeze(axis=2)
        return _clahe(image)

    # Proceed with color image
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = _clahe(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    except cv2.error:
        # Fallback for non-BGR images (e.g., RGB)
        print("Warning: CLAHE BGR-to-LAB conversion failed. Applying to channels individually.")
        return _channelwise(_clahe, image)
