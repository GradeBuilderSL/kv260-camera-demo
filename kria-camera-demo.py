#!/usr/bin/env python3
"""Kria Camera Demo - Intel RealSense Camera Frame Capture and Visualization"""

import pyrealsense2 as rs
import numpy as np
import cv2
from pynq_dpu import DpuOverlay
import re
import os
import sys
import time
import subprocess
import threading
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
        self.labels = KriaCameraDemo.load_labels(labels_path)
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

        # Platform stats monitoring
        self.platform_stats = {}
        self.platform_stats_lock = threading.Lock()
        self.platform_stats_interval = platform_stats_interval
        self.platform_stats_thread = None
        self.platform_stats_running = False

    def start_camera(self):
        """Start the RealSense camera pipeline."""
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print("Starting camera stream...")
        self.pipeline.start(self.config)

    def parse_platform_stats(self, output):
        """Parse xmutil xlnx_platformstats output.

        Args:
            output: String output from xmutil xlnx_platformstats command

        Returns:
            dict: Dictionary containing parsed platform statistics
        """
        stats = {}
        lines = output.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Parse SOM power metrics
            if 'SOM total power' in line:
                match = re.search(r':\s*(\d+)\s*mW', line)
                if match:
                    stats['power_mw'] = int(match.group(1))
            elif 'SOM total current' in line:
                match = re.search(r':\s*(\d+)\s*mA', line)
                if match:
                    stats['current_ma'] = int(match.group(1))
            elif 'SOM total voltage' in line:
                match = re.search(r':\s*(\d+)\s*mV', line)
                if match:
                    stats['voltage_mv'] = int(match.group(1))

            # Parse CPU utilization
            elif line.startswith('CPU') and '%' in line:
                match = re.search(r'CPU(\d+)\s*:\s*([\d.]+)%', line)
                if match:
                    cpu_num = int(match.group(1))
                    cpu_util = float(match.group(2))
                    stats[f'cpu{cpu_num}_util'] = cpu_util

            # Parse temperatures
            elif 'LPD temperature' in line:
                match = re.search(r':\s*(\d+)\s*C', line)
                if match:
                    stats['lpd_temp_c'] = int(match.group(1))
            elif 'FPD temperature' in line:
                match = re.search(r':\s*(\d+)\s*C', line)
                if match:
                    stats['fpd_temp_c'] = int(match.group(1))
            elif 'PL temperature' in line:
                match = re.search(r':\s*(\d+)\s*C', line)
                if match:
                    stats['pl_temp_c'] = int(match.group(1))

            # Parse memory utilization
            elif line.startswith('MemTotal'):
                match = re.search(r':\s*(\d+)\s*kB', line)
                if match:
                    stats['mem_total_kb'] = int(match.group(1))
            elif line.startswith('MemAvailable'):
                match = re.search(r':\s*(\d+)\s*kB', line)
                if match:
                    stats['mem_available_kb'] = int(match.group(1))

        return stats

    def update_platform_stats(self):
        """Fetch and update platform statistics by calling xmutil xlnx_platformstats."""
        try:
            result = subprocess.run(
                ['sudo', 'xmutil', 'xlnx_platformstats'],
                capture_output=True,
                text=True,
                timeout=5.0
            )

            if result.returncode == 0:
                stats = self.parse_platform_stats(result.stdout)
                with self.platform_stats_lock:
                    self.platform_stats = stats
            else:
                print(f"Error running xmutil: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("xmutil command timed out")
        except Exception as e:
            print(f"Error updating platform stats: {e}")

    def platform_stats_monitor_loop(self):
        """Background thread loop to periodically update platform stats."""
        print(f"Starting platform stats monitor (interval: {self.platform_stats_interval}s)")
        while self.platform_stats_running:
            self.update_platform_stats()
            time.sleep(self.platform_stats_interval)

    def start_platform_stats_monitor(self):
        """Start the platform stats monitoring thread."""
        if not self.platform_stats_running:
            self.platform_stats_running = True
            self.platform_stats_thread = threading.Thread(
                target=self.platform_stats_monitor_loop,
                daemon=True
            )
            self.platform_stats_thread.start()

    def stop_platform_stats_monitor(self):
        """Stop the platform stats monitoring thread."""
        if self.platform_stats_running:
            self.platform_stats_running = False
            if self.platform_stats_thread:
                self.platform_stats_thread.join(timeout=2.0)

    def get_platform_stats(self):
        """Get current platform stats in a thread-safe manner.

        Returns:
            dict: Copy of current platform statistics
        """
        with self.platform_stats_lock:
            return self.platform_stats.copy()

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

    def make_prediction(self, color_image):
        """Run inference on the input image using DPU.

        Args:
            color_image: Input color image

        Returns:
            tuple: (predictions, inference_time) where predictions is a list of
                   tuples (class_index, confidence) and inference_time is in seconds
        """
        # Apply normalization to color image
        normalized_image = KriaCameraDemo.preprocess_fn(
            color_image,
            self.norm_params['mean'],
            self.norm_params['scale']
        )

        image = self.input_data[0]
        image[0,...] = normalized_image.reshape(self.shapeIn[1:])

        # Measure DPU execution time
        start_time = time.perf_counter()
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.perf_counter()
        inference_time = end_time - start_time

        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        softmax = KriaCameraDemo.calculate_softmax(temp[0][0])

        predictions = KriaCameraDemo.predict_label(softmax, top_k=5)
        return predictions, inference_time

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
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

    @staticmethod
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

    @staticmethod
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
        luminance = KriaCameraDemo.calculate_luminance(mean_color)

        # Use white text on dark background, black text on light background
        # Threshold at 128 (middle of 0-255 range)
        return (255, 255, 255) if luminance < 128 else (0, 0, 0)

    @staticmethod
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

    def cleanup(self):
        """Release all resources."""
        # Stop platform stats monitoring
        self.stop_platform_stats_monitor()

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
        dpu_bit_path="dpu.bit",
        labels_path="words.txt"
    )

    try:
        demo.start_camera()
        demo.start_platform_stats_monitor()

        print("Press 'q' to quit")

        # Variables for FPS calculation
        prev_frame_time = time.perf_counter()

        while True:
            # Get frames from camera
            depth_image, color_image = demo.get_frames()

            if depth_image is None or color_image is None:
                continue

            # Calculate actual FPS
            current_frame_time = time.perf_counter()
            frame_delta = current_frame_time - prev_frame_time
            actual_fps = 1.0 / frame_delta if frame_delta > 0 else 0
            prev_frame_time = current_frame_time

            # Run prediction - returns top-5 predictions and inference time
            predictions, inference_time = demo.predict(color_image)

            # Calculate DPU inference FPS
            dpu_fps = 1.0 / inference_time if inference_time > 0 else 0

            # Apply colormap on depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_WINTER
            )

            # Add performance info and top-5 predictions to color image
            display_image = color_image.copy()

            # Display actual FPS and DPU inference time at the top
            perf_text = f"FPS: {actual_fps:.1f} | DPU: {inference_time*1000:.1f}ms ({dpu_fps:.1f} FPS)"
            perf_font = cv2.FONT_HERSHEY_SIMPLEX
            perf_scale = 0.6
            perf_thickness = 2
            perf_pos = (10, 20)

            # Get text size for contrast calculation
            (perf_width, perf_height), _ = cv2.getTextSize(
                perf_text, perf_font, perf_scale, perf_thickness
            )

            # Determine contrasting color based on background
            perf_color = KriaCameraDemo.get_contrasting_color(
                display_image, perf_pos[0], perf_pos[1] - perf_height,
                perf_width, perf_height
            )

            # Draw performance text with semi-transparent background
            demo.draw_text_with_background(
                display_image, perf_text, perf_pos, perf_font,
                perf_scale, perf_color, perf_thickness, bg_opacity=0.5
            )

            # Display "Classification" section header
            y_offset = 50
            header_font = cv2.FONT_HERSHEY_SIMPLEX
            header_scale = 0.6
            header_thickness = 2
            header_text = "Classification"
            header_pos = (10, y_offset)

            # Get text size for contrast calculation
            (header_width, header_height), _ = cv2.getTextSize(
                header_text, header_font, header_scale, header_thickness
            )

            # Determine contrasting color based on background
            header_color = KriaCameraDemo.get_contrasting_color(
                display_image, header_pos[0], header_pos[1] - header_height,
                header_width, header_height
            )

            # Draw section header with semi-transparent background
            demo.draw_text_with_background(
                display_image, header_text, header_pos, header_font,
                header_scale, header_color, header_thickness, bg_opacity=0.5
            )

            # Display top-5 predictions
            y_offset += 30
            pred_font = cv2.FONT_HERSHEY_SIMPLEX
            pred_scale = 0.5
            pred_thickness = 1

            for i, (label_idx, confidence) in enumerate(predictions, 1):
                label_text = demo.get_label_text(label_idx)
                # Format: "1. [152] 95.2% - Chihuahua"
                text = f"{i}. [{label_idx}] {confidence:.1%} - {label_text[:30]}"
                pred_pos = (10, y_offset)

                # Get text size for contrast calculation
                (text_width, text_height), _ = cv2.getTextSize(
                    text, pred_font, pred_scale, pred_thickness
                )

                # Determine contrasting color based on background
                text_color = KriaCameraDemo.get_contrasting_color(
                    display_image, pred_pos[0], pred_pos[1] - text_height,
                    text_width, text_height
                )

                # Draw prediction text with semi-transparent background
                demo.draw_text_with_background(
                    display_image, text, pred_pos, pred_font,
                    pred_scale, text_color, pred_thickness, bg_opacity=0.5
                )

                y_offset += 25

            # Display platform stats
            platform_stats = demo.get_platform_stats()
            if platform_stats:
                y_offset += 10  # Add some spacing

                # Display "System Info" section header
                header_font = cv2.FONT_HERSHEY_SIMPLEX
                header_scale = 0.6
                header_thickness = 2
                header_text = "System Info"
                header_pos = (10, y_offset)

                # Get text size for contrast calculation
                (header_width, header_height), _ = cv2.getTextSize(
                    header_text, header_font, header_scale, header_thickness
                )

                # Determine contrasting color based on background
                header_color = KriaCameraDemo.get_contrasting_color(
                    display_image, header_pos[0], header_pos[1] - header_height,
                    header_width, header_height
                )

                # Draw section header with semi-transparent background
                demo.draw_text_with_background(
                    display_image, header_text, header_pos, header_font,
                    header_scale, header_color, header_thickness, bg_opacity=0.5
                )

                y_offset += 25
                stats_font = cv2.FONT_HERSHEY_SIMPLEX
                stats_scale = 0.5
                stats_thickness = 1

                # Prepare stats text lines
                stats_lines = []

                # Power metrics
                if 'power_mw' in platform_stats:
                    stats_lines.append(f"Power: {platform_stats['power_mw']} mW")
                if 'current_ma' in platform_stats:
                    stats_lines.append(f"Current: {platform_stats['current_ma']} mA")
                if 'voltage_mv' in platform_stats:
                    stats_lines.append(f"Voltage: {platform_stats['voltage_mv']} mV")

                # Temperature metrics
                temps = []
                if 'lpd_temp_c' in platform_stats:
                    temps.append(f"LPD:{platform_stats['lpd_temp_c']}C")
                if 'fpd_temp_c' in platform_stats:
                    temps.append(f"FPD:{platform_stats['fpd_temp_c']}C")
                if 'pl_temp_c' in platform_stats:
                    temps.append(f"PL:{platform_stats['pl_temp_c']}C")
                if temps:
                    stats_lines.append(f"Temp: {' '.join(temps)}")

                # CPU utilization
                cpu_utils = []
                for i in range(4):
                    key = f'cpu{i}_util'
                    if key in platform_stats:
                        cpu_utils.append(f"{platform_stats[key]:.1f}%")
                if cpu_utils:
                    stats_lines.append(f"CPU: {' '.join(cpu_utils)}")

                # Memory utilization
                if 'mem_total_kb' in platform_stats and 'mem_available_kb' in platform_stats:
                    mem_used_mb = (platform_stats['mem_total_kb'] - platform_stats['mem_available_kb']) / 1024
                    mem_total_mb = platform_stats['mem_total_kb'] / 1024
                    mem_percent = (mem_used_mb / mem_total_mb) * 100 if mem_total_mb > 0 else 0
                    stats_lines.append(f"RAM: {mem_used_mb:.0f}/{mem_total_mb:.0f} MB ({mem_percent:.1f}%)")

                # Draw each stats line
                for stats_text in stats_lines:
                    stats_pos = (10, y_offset)

                    # Get text size for contrast calculation
                    (text_width, text_height), _ = cv2.getTextSize(
                        stats_text, stats_font, stats_scale, stats_thickness
                    )

                    # Determine contrasting color based on background
                    text_color = KriaCameraDemo.get_contrasting_color(
                        display_image, stats_pos[0], stats_pos[1] - text_height,
                        text_width, text_height
                    )

                    # Draw stats text with semi-transparent background
                    demo.draw_text_with_background(
                        display_image, stats_text, stats_pos, stats_font,
                        stats_scale, text_color, stats_thickness, bg_opacity=0.5
                    )

                    y_offset += 20

            images = display_image

            # Show images
            cv2.imshow('RealSense - Color', images)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
