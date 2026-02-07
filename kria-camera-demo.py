#!/usr/bin/env python3
"""Kria Camera Demo - Intel RealSense Camera Frame Capture and Visualization"""

import pyrealsense2 as rs
import numpy as np
import cv2
from pynq_dpu import DpuOverlay
import re
import os
import sys
from pathlib import Path


def check_and_elevate_privileges():
    """Check if running as root and re-execute with sudo if needed.

    PYNQ DPU overlay requires root permissions to access hardware MMIO registers.
    """
    if os.geteuid() != 0:
        print("Root privileges required for PYNQ DPU hardware access.")
        print("Re-executing with sudo...")

        # Re-execute the script with sudo
        args = ['sudo', sys.executable] + sys.argv
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




def predict_label(softmax):

    return np.argmax(softmax) - 1




class KriaCameraDemo:
    """Main class for Kria camera demo with DPU inference."""

    def __init__(self, prototxt_path, model_path, dpu_bit_path):
        """Initialize the camera demo with DPU model and normalization parameters.

        Args:
            prototxt_path: Path to the prototxt file with normalization parameters
            model_path: Path to the DPU xmodel file
            dpu_bit_path: Path to the DPU bitstream file
        """
        # Load normalization parameters
        print(f"Loading normalization parameters from {prototxt_path}...")
        self.norm_params = parse_prototxt(prototxt_path)
        print(f"Mean values (BGR): {self.norm_params['mean']}")
        print(f"Scale values (BGR): {self.norm_params['scale']}")

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

    def start_camera(self):
        """Start the RealSense camera pipeline."""
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print("Starting camera stream...")
        self.pipeline.start(self.config)

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
        result = np.exp(data)
        return result

    def make_prediction(self, color_image):
        """Run inference on the input image using DPU.

        Args:
            color_image: Input color image

        Returns:
            int: Predicted label
        """
        # Apply normalization to color image
        normalized_image = KriaCameraDemo.preprocess_fn(
            color_image,
            self.norm_params['mean'],
            self.norm_params['scale']
        )

        image = self.input_data[0]
        image[0,...] = normalized_image.reshape(self.shapeIn[1:])

        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)

        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        softmax = KriaCameraDemo.calculate_softmax(temp[0][0])

        label = predict_label(softmax)
        return label

    @staticmethod
    def resize_shortest_edge(image, size):
        H, W = image.shape[:2]
        if H >= W:
            nW = size
            nH = int(float(H)/W * size)
        else:
            nH = size
            nW = int(float(W)/H * size)
        return cv2.resize(image,(nW,nH))

    @staticmethod
    def central_crop(image, crop_height, crop_width):
        image_height = image.shape[0]
        image_width = image.shape[1]
        offset_height = (image_height - crop_height) // 2
        offset_width = (image_width - crop_width) // 2
        return image[offset_height:offset_height + crop_height, offset_width:
                    offset_width + crop_width, :]

    @staticmethod
    def preprocess_fn(image, means, scales, crop_height = 224, crop_width = 224):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = KriaCameraDemo.resize_shortest_edge(image, 256)
        image = KriaCameraDemo.normalize_image(image, means, scales)
        image = KriaCameraDemo.central_crop(image, crop_height, crop_width)
        return image

    @staticmethod
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

    def predict(self, color_image):
        """Run inference on the color image.

        Args:
            color_image: Input color image (BGR format)

        Returns:
            int: Predicted label
        """
        return self.make_prediction(color_image)

    def cleanup(self):
        """Release all resources."""
        self.pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stream stopped")

        if self.dpu:
            del self.dpu

        self.overlay.free()
        print("DPU overlay resources released")


def main():
    # Check for root privileges and elevate if needed
    check_and_elevate_privileges()

    # Initialize the camera demo
    demo = KriaCameraDemo(
        prototxt_path=Path("models/mobilenet_v2/mobilenet_v2.prototxt"),
        model_path="./models/mobilenet_v2/mobilenet_v2.xmodel",
        dpu_bit_path="dpu.bit"
    )

    try:
        demo.start_camera()

        print("Press 'q' to quit")
        while True:
            # Get frames from camera
            depth_image, color_image = demo.get_frames()

            if depth_image is None or color_image is None:
                continue

            # Run prediction
            label = demo.predict(color_image)
            print(f"Predicted label: {label}")

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_WINTER
            )

            # Stack images horizontally: color | depth
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.imshow('RealSense - Color | Depth', images)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
