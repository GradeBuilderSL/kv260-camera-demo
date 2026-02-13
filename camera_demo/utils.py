#!/usr/bin/env python3
"""Utility functions for Kria camera demo."""

import os
import sys
import re
import json
from pathlib import Path


def check_and_elevate_privileges():
    """Check if running as root and re-execute with sudo if needed.

    PYNQ DPU overlay requires root permissions to access hardware MMIO registers.
    """
    if os.geteuid() != 0:
        print("Root privileges required for PYNQ DPU hardware access.")
        print("Re-executing with sudo...")

        # Re-execute the script with sudo
        args = ['sudo', '-E', sys.executable] + sys.argv
        os.execvp('sudo', args)


def parse_prototxt(prototxt_path):
    """Parse MobileNet V2 prototxt file to extract normalization parameters.

    Args:
        prototxt_path: Path to the .prototxt configuration file

    Returns:
        dict: Dictionary containing 'mean' (list of 3 floats) and 'scale' (list of 3 floats)
    """
    with open(prototxt_path, 'r') as f:
        content = f.read()

    # Extract mean values (RGB order for Caffe)
    mean_pattern = r'mean:\s*([\d.]+)'
    mean_values = [float(m) for m in re.findall(mean_pattern, content)]

    # Extract scale values
    scale_pattern = r'scale:\s*([\d.]+)'
    scale_values = [float(s) for s in re.findall(scale_pattern, content)]

    if len(mean_values) != 3 or len(scale_values) != 3:
        raise ValueError(f"Expected 3 mean and 3 scale values, got {len(mean_values)} means and {len(scale_values)} scales")

    return {
        'mean': mean_values,
        'scale': scale_values
    }


def load_labels(labels_path):
    """Load ImageNet class labels from words.txt file.

    Args:
        labels_path: Path to the words.txt file containing class labels

    Returns:
        list: List of label strings, indexed by class ID
    """
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                labels.append(line)
    return labels


def load_model_metadata(model_folder):
    """Load model metadata from meta.json file.

    Args:
        model_folder: Path to the folder containing meta.json

    Returns:
        dict: Dictionary containing metadata with keys:
            - 'model_path': Full path to the .xmodel file
            - 'prototxt_path': Full path to the .prototxt file
            - 'metadata': Complete metadata from meta.json

    Raises:
        FileNotFoundError: If meta.json doesn't exist in the folder
        KeyError: If 'filename' field is missing from meta.json
    """
    model_folder = Path(model_folder)
    meta_json_path = model_folder / "meta.json"

    if not meta_json_path.exists():
        raise FileNotFoundError(f"meta.json not found in {model_folder}")

    # Read and parse meta.json
    with open(meta_json_path, 'r') as f:
        metadata = json.load(f)

    # Extract model filename
    if 'filename' not in metadata:
        raise KeyError("'filename' field not found in meta.json")

    model_filename = metadata['filename']
    model_path = model_folder / model_filename

    # Construct prototxt path (same base name with .prototxt extension)
    model_base_name = Path(model_filename).stem
    prototxt_path = model_folder / f"{model_base_name}.prototxt"

    return {
        'model_path': str(model_path),
        'prototxt_path': str(prototxt_path),
        'metadata': metadata
    }
