#!/usr/bin/env python3

############################################################################
# Copyright 2026 GradeBuilder SL                                           #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

"""Image preprocessing functions for DPU inference."""

import cv2
import numpy as np


def resize_shortest_edge(image, size):
    """Resize image such that the shortest edge equals the target size.

    Args:
        image: Input image
        size: Target size for shortest edge

    Returns:
        numpy.ndarray: Resized image
    """
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image, (nW, nH))


def central_crop(image, crop_height, crop_width):
    """Crop the center region of an image.

    Args:
        image: Input image
        crop_height: Height of crop region
        crop_width: Width of crop region

    Returns:
        numpy.ndarray: Cropped image
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:
                offset_width + crop_width, :]


def normalize_image(image, mean, scale):
    """Normalize image using mean subtraction and scaling.

    Args:
        image: Input image as numpy array (BGR format)
        mean: List of 3 mean values for BGR channels
        scale: List of 3 scale values for BGR channels

    Returns:
        numpy.ndarray: Normalized image (float32)
    """
    # Convert to float32 for normalization
    normalized = image.astype(np.float32)

    # Apply mean subtraction and scaling per channel (BGR order)
    for i in range(3):
        normalized[:, :, i] = (normalized[:, :, i] - mean[i]) * scale[i]

    return normalized


def preprocess_fn(image, means, scales, crop_height=224, crop_width=224):
    """Complete preprocessing pipeline for MobileNet V2 input.

    Args:
        image: Input image (BGR format)
        means: List of 3 mean values for BGR channels
        scales: List of 3 scale values for BGR channels
        crop_height: Height of final crop (default: 224)
        crop_width: Width of final crop (default: 224)

    Returns:
        numpy.ndarray: Preprocessed image ready for DPU inference
    """
    image = resize_shortest_edge(image, 256)
    image = normalize_image(image, means, scales)
    image = central_crop(image, crop_height, crop_width)
    return image
