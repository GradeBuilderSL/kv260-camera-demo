#!/usr/bin/env python3
"""Kria Camera Demo package - Modular components for RealSense camera with DPU inference."""

from .kria_camera_demo import KriaCameraDemo
from .utils import check_and_elevate_privileges, parse_prototxt, load_labels, load_model_metadata
from .preprocessing import preprocess_fn, resize_shortest_edge, central_crop, normalize_image
from .visualization import get_contrasting_color, draw_text_with_background, calculate_luminance
from .platform_monitor import PlatformMonitor

__all__ = [
    'KriaCameraDemo',
    'check_and_elevate_privileges',
    'parse_prototxt',
    'load_labels',
    'load_model_metadata',
    'preprocess_fn',
    'resize_shortest_edge',
    'central_crop',
    'normalize_image',
    'get_contrasting_color',
    'draw_text_with_background',
    'calculate_luminance',
    'PlatformMonitor',
]
