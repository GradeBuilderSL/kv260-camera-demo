#!/usr/bin/env python3
"""Kria Camera Demo - Main application class for RealSense camera with DPU inference."""

import pyrealsense2 as rs
import numpy as np
import cv2
from pynq_dpu import DpuOverlay
import time

from .utils import parse_prototxt, load_labels
from .preprocessing import preprocess_fn
from .platform_monitor import PlatformMonitor
from .visualization import get_contrasting_color, draw_text_with_background


class KriaCameraDemo:
    """Main class for Kria camera demo with DPU inference."""

    def __init__(self, prototxt_path, model_path, dpu_bit_path, labels_path,
                 platform_stats_interval=2.0):
        """Initialize the camera demo with DPU model and normalization parameters.

        Args:
            prototxt_path: Path to the prototxt file with normalization parameters
            model_path: Path to the DPU xmodel file
            dpu_bit_path: Path to the DPU bitstream file
            labels_path: Path to the words.txt file with class labels
            platform_stats_interval: Interval in seconds to update platform stats (default: 2.0)
        """
        # Load normalization parameters
        print(f"Loading normalization parameters from {prototxt_path}...")
        self.norm_params = parse_prototxt(prototxt_path)
        print(f"Mean values (BGR): {self.norm_params['mean']}")
        print(f"Scale values (BGR): {self.norm_params['scale']}")

        # Load class labels
        print(f"Loading class labels from {labels_path}...")
        self.labels = load_labels(labels_path)
        print(f"Loaded {len(self.labels)} class labels")

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Configure streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Initialize DPU
        print("Creating DPU overlay...")
        self.overlay = DpuOverlay(dpu_bit_path)
        self.overlay.load_model(model_path)

        self.dpu = self.overlay.runner
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut = tuple(outputTensors[0].dims)
        self.outputSize = int(outputTensors[0].get_data_size() / self.shapeIn[0])

        self.output_data = [np.empty(self.shapeOut, dtype=np.float32, order="C")]
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]

        # Initialize platform monitor
        self.platform_monitor = PlatformMonitor(update_interval=platform_stats_interval)

    def start_camera(self):
        """Start the RealSense camera pipeline."""
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print("Starting camera stream...")
        self.pipeline.start(self.config)

    def start_platform_monitor(self):
        """Start the platform stats monitoring thread."""
        self.platform_monitor.start()

    def stop_platform_monitor(self):
        """Stop the platform stats monitoring thread."""
        self.platform_monitor.stop()

    def get_platform_stats(self):
        """Get current platform stats.

        Returns:
            dict: Copy of current platform statistics
        """
        return self.platform_monitor.get_stats()

    def get_frames(self):
        """Get depth and color frames from the camera.

        Returns:
            tuple: (depth_image, color_image) as numpy arrays, or (None, None) if frames not available
        """
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    @staticmethod
    def calculate_softmax(data):
        """Calculate softmax probabilities from logits.

        Args:
            data: Raw logits from model output

        Returns:
            numpy.ndarray: Normalized softmax probabilities (sum to 1.0)
        """
        # Subtract max for numerical stability
        exp_data = np.exp(data - np.max(data))
        return exp_data / np.sum(exp_data)

    @staticmethod
    def predict_label(softmax, top_k=5):
        """Get top-k predictions from softmax output.

        Args:
            softmax: Softmax probability array
            top_k: Number of top predictions to return

        Returns:
            list: List of tuples (class_index, confidence) sorted by confidence descending
        """
        # Get top-k indices sorted by confidence (descending)
        top_indices = np.argsort(softmax)[-top_k:][::-1]

        # Get corresponding confidence values
        top_confidences = softmax[top_indices]

        # Return list of (index, confidence) tuples
        return [(int(idx), float(conf)) for idx, conf in zip(top_indices, top_confidences)]

    def make_prediction(self, color_image):
        """Run inference on the input image using DPU.

        Args:
            color_image: Input color image

        Returns:
            tuple: (predictions, inference_time) where predictions is a list of
                   tuples (class_index, confidence) and inference_time is in seconds
        """
        # Apply normalization to color image
        normalized_image = preprocess_fn(
            color_image,
            self.norm_params['mean'],
            self.norm_params['scale']
        )

        image = self.input_data[0]
        image[0, ...] = normalized_image.reshape(self.shapeIn[1:])

        # Measure DPU execution time
        start_time = time.perf_counter()
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        softmax = self.calculate_softmax(temp[0][0])

        predictions = self.predict_label(softmax, top_k=5)
        return predictions, inference_time

    def predict(self, color_image):
        """Run inference on the color image.

        Args:
            color_image: Input color image (BGR format)

        Returns:
            tuple: (predictions, inference_time) where predictions is a list of
                   tuples (class_index, confidence) and inference_time is in seconds
        """
        return self.make_prediction(color_image)

    def get_label_text(self, label_index):
        """Get the text label for a given class index.

        Args:
            label_index: Integer class index (0-999)

        Returns:
            str: Text label or "Unknown" if index is out of range
        """
        if 0 <= label_index < len(self.labels):
            return self.labels[label_index]
        return f"Unknown (index {label_index})"

    def cleanup(self):
        """Release all resources."""
        # Stop platform stats monitoring
        self.stop_platform_monitor()

        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stream stopped")

        if self.dpu:
            del self.dpu

        self.overlay.free()
        print("DPU overlay resources released")
