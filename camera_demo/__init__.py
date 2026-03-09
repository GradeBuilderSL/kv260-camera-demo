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
