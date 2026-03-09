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

"""Visualization helper functions for drawing text and overlays."""

import cv2


def calculate_luminance(bgr_color):
    """Calculate relative luminance of a BGR color using ITU-R BT.709 standard.

    Args:
        bgr_color: Tuple or array of (B, G, R) values in range [0, 255]

    Returns:
        float: Relative luminance in range [0, 255]
    """
    # ITU-R BT.709 coefficients for RGB (convert from BGR)
    b, g, r = bgr_color[0], bgr_color[1], bgr_color[2]
    # Weighted sum: 0.2126*R + 0.7152*G + 0.0722*B
    return 0.0722 * b + 0.7152 * g + 0.2126 * r


def get_contrasting_color(image, x, y, width, height):
    """Determine optimal text color (white or black) based on background luminance.

    Args:
        image: Input BGR image
        x, y: Top-left corner of text region
        width, height: Dimensions of text region

    Returns:
        tuple: BGR color tuple (255, 255, 255) for white or (0, 0, 0) for black
    """
    img_h, img_w = image.shape[:2]

    # Clamp coordinates to image bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + width)
    y2 = min(img_h, y + height)

    # Extract region of interest
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        # Default to white if region is invalid
        return (255, 255, 255)

    # Calculate mean color of the region
    mean_color = cv2.mean(roi)[:3]  # Get BGR values only

    # Calculate luminance
    luminance = calculate_luminance(mean_color)

    # Use white text on dark background, black text on light background
    # Threshold at 128 (middle of 0-255 range)
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def draw_text_with_background(image, text, position, font, font_scale,
                              text_color, thickness, bg_opacity=0.6):
    """Draw text with a semi-transparent background for improved readability.

    Args:
        image: Image to draw on
        text: Text string to draw
        position: (x, y) position for text
        font: OpenCV font type
        font_scale: Font scale factor
        text_color: BGR color tuple for text
        thickness: Text thickness
        bg_opacity: Background opacity (0=transparent, 1=opaque)

    Returns:
        numpy.ndarray: Image with text drawn
    """
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position
    padding = 5

    # Define background rectangle
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding

    # Create semi-transparent background
    overlay = image.copy()

    # Use inverse of text color for background (with some adjustment)
    bg_color = tuple(255 - c for c in text_color)

    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)

    # Blend overlay with original image
    cv2.addWeighted(overlay, bg_opacity, image, 1 - bg_opacity, 0, image)

    # Draw text on top
    cv2.putText(image, text, position, font, font_scale, text_color, thickness)

    return image
