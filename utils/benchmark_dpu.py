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

"""Synthetic DPU Benchmark - Test DPU inference performance with random data."""

import logging
import sys
import numpy as np
import time
import argparse
import asyncio
from pathlib import Path

from camera_demo import (
    check_and_elevate_privileges,
    parse_prototxt,
    load_labels,
    load_model_metadata,
    preprocess_fn
)

from pynq_dpu import DpuOverlay

# ---------------------------------------------------------------------------
# Logging — dmesg-style timestamps (milliseconds since boot)
# ---------------------------------------------------------------------------

def _read_uptime_ms() -> float:
    """Read current uptime from /proc/uptime and return it in milliseconds."""
    with open("/proc/uptime") as f:
        return float(f.read().split()[0]) * 1000.0


class _BootTimeFormatter(logging.Formatter):
    """Logging formatter that prefixes every record with time-since-boot in ms.

    Format matches dmesg style:
        [305122460.123] INFO     message text
                                 continuation line (aligned)
    The uptime is read once from /proc/uptime at construction and then tracked
    cheaply via time.monotonic() to avoid a file read on every log call.
    """

    def __init__(self) -> None:
        super().__init__()
        self._uptime_at_init_ms = _read_uptime_ms()
        self._mono_at_init      = time.monotonic()

    def _boot_ms(self) -> float:
        return self._uptime_at_init_ms + (time.monotonic() - self._mono_at_init) * 1000.0

    def format(self, record: logging.LogRecord) -> str:
        ms     = self._boot_ms()
        prefix = f"[{ms:>13.3f}] {record.levelname:<8} "
        msg    = record.getMessage()
        indent = " " * len(prefix)
        return prefix + ("\n" + indent).join(msg.splitlines())


def _setup_logging(level: int = logging.DEBUG) -> None:
    handler = logging.StreamHandler(sys.stdout)   # stdout so tee captures it
    handler.setFormatter(_BootTimeFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


_setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vaitrace instrumentation
# ---------------------------------------------------------------------------

# Import vaitrace for detailed profiling
try:
    from vaitrace_py import vai_tracepoint
    VAITRACE_AVAILABLE = True
    logger.info("vaitrace instrumentation enabled")
except ImportError:
    # Fallback: create dummy decorator
    def vai_tracepoint(func):
        return func
    VAITRACE_AVAILABLE = False
    logger.warning("vaitrace not available, running without instrumentation")


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="DPU Synthetic Benchmark - Test inference with random data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m", "--model-dir",
        type=str,
        default="models/mobilenet_v2",
        help="Path to the model directory containing meta.json"
    )

    parser.add_argument(
        "-b", "--dpu-bit",
        type=str,
        default="dpu.bit",
        help="Path to the DPU bitstream file"
    )

    parser.add_argument(
        "-l", "--labels",
        type=str,
        default="words.txt",
        help="Path to the class labels file"
    )

    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=100,
        help="Number of frames to process"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before measurement"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results"
    )

    return parser.parse_args()


class DpuBenchmark:
    """Synthetic benchmark class for DPU inference testing."""

    def __init__(self, prototxt_path, model_path, dpu_bit_path, labels_path):
        """Initialize the benchmark with DPU model and normalization parameters.

        Args:
            prototxt_path: Path to the prototxt file with normalization parameters
            model_path: Path to the DPU xmodel file
            dpu_bit_path: Path to the DPU bitstream file
            labels_path: Path to the words.txt file with class labels
        """
        # Load normalization parameters
        logger.info(f"Loading normalization parameters from {prototxt_path}")
        self.norm_params = parse_prototxt(prototxt_path)
        logger.debug(f"Mean values (BGR): {self.norm_params['mean']}")
        logger.debug(f"Scale values (BGR): {self.norm_params['scale']}")

        # Load class labels
        logger.info(f"Loading class labels from {labels_path}")
        self.labels = load_labels(labels_path)
        logger.info(f"Loaded {len(self.labels)} class labels")

        # Ensure event loop exists for PYNQ (needed for vaitrace profiling)
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug("Created new asyncio event loop for PYNQ")

        # Initialize DPU
        logger.info("Creating DPU overlay...")
        self.overlay = DpuOverlay(dpu_bit_path)
        self.overlay.load_model(model_path)

        self.dpu = self.overlay.runner
        inputTensors = self.dpu.get_input_tensors()
        outputTensors = self.dpu.get_output_tensors()

        self.shapeIn = tuple(inputTensors[0].dims)
        self.shapeOut = tuple(outputTensors[0].dims)
        self.outputSize = int(outputTensors[0].get_data_size() / self.shapeIn[0])

        logger.info(f"DPU Input shape:  {self.shapeIn}")
        logger.info(f"DPU Output shape: {self.shapeOut}")

        self.output_data = [np.empty(self.shapeOut, dtype=np.float32, order="C")]
        self.input_data = [np.empty(self.shapeIn, dtype=np.float32, order="C")]

    def generate_random_image(self, height=480, width=640):
        """Generate a random BGR image.

        Args:
            height: Image height
            width: Image width

        Returns:
            numpy.ndarray: Random BGR image (height, width, 3)
        """
        # Generate random uint8 values for a BGR image
        return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

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

    @vai_tracepoint
    def preprocess_image(self, color_image):
        """Preprocess image for DPU inference."""
        normalized_image = preprocess_fn(
            color_image,
            self.norm_params['mean'],
            self.norm_params['scale']
        )
        image = self.input_data[0]
        image[0, ...] = normalized_image.reshape(self.shapeIn[1:])

    @vai_tracepoint
    def run_dpu_inference(self):
        """Execute DPU inference."""
        start_time = time.perf_counter()
        job_id = self.dpu.execute_async(self.input_data, self.output_data)
        self.dpu.wait(job_id)
        end_time = time.perf_counter()
        return end_time - start_time

    @vai_tracepoint
    def postprocess_output(self):
        """Postprocess DPU output to get predictions."""
        temp = [j.reshape(1, self.outputSize) for j in self.output_data]
        softmax = self.calculate_softmax(temp[0][0])
        predictions = self.predict_label(softmax, top_k=5)
        return predictions

    def run_inference(self, color_image):
        """Run inference on the input image using DPU.

        Args:
            color_image: Input color image (BGR format, uint8)

        Returns:
            tuple: (predictions, inference_time) where predictions is a list of
                   tuples (class_index, confidence) and inference_time is in seconds
        """
        # Preprocessing with tracepoint
        self.preprocess_image(color_image)

        # DPU inference with tracepoint
        inference_time = self.run_dpu_inference()

        # Postprocessing with tracepoint
        predictions = self.postprocess_output()

        return predictions, inference_time

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
        if self.dpu:
            del self.dpu

        self.overlay.free()
        logger.info("DPU overlay resources released")


def main():
    """Main benchmark loop."""
    # Parse command-line arguments
    args = parse_arguments()

    # Check for root privileges and elevate if needed
    check_and_elevate_privileges()

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load model metadata from meta.json
    logger.info(f"Loading model metadata from {args.model_dir}/meta.json")
    model_info = load_model_metadata(args.model_dir)

    logger.debug(f"Model file:    {model_info['model_path']}")
    logger.debug(f"Prototxt file: {model_info['prototxt_path']}")
    logger.debug(f"Target:        {model_info['metadata'].get('target', 'Unknown')}")

    # Initialize the benchmark
    benchmark = DpuBenchmark(
        prototxt_path=model_info['prototxt_path'],
        model_path=model_info['model_path'],
        dpu_bit_path=args.dpu_bit,
        labels_path=args.labels
    )

    try:
        logger.info("=" * 60)
        logger.info("DPU SYNTHETIC BENCHMARK")
        logger.info(f"Warmup iterations:    {args.warmup}")
        logger.info(f"Benchmark iterations: {args.num_frames}")
        logger.info("=" * 60)

        # Warmup phase
        logger.info("Warming up DPU...")
        for i in range(args.warmup):
            random_image = benchmark.generate_random_image()
            _, warmup_time = benchmark.run_inference(random_image)
            if (i + 1) % 5 == 0:
                logger.debug(f"Warmup {i+1}/{args.warmup} — {warmup_time*1000:.2f} ms")

        logger.info("Warmup complete. Starting benchmark...")

        # Benchmark phase
        inference_times = []
        start_time = time.perf_counter()

        for i in range(args.num_frames):
            # Generate random image
            random_image = benchmark.generate_random_image()

            # Run inference (uses decorated methods with tracepoints)
            predictions, inference_time = benchmark.run_inference(random_image)
            inference_times.append(inference_time)

            # Print progress
            if (i + 1) % 10 == 0:
                avg_time = np.mean(inference_times)
                logger.info(
                    f"Processed {i+1}/{args.num_frames} frames — "
                    f"avg DPU: {avg_time*1000:.2f} ms ({1.0/avg_time:.2f} FPS)"
                )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Calculate statistics
        inference_times = np.array(inference_times)
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        p50_time = np.percentile(inference_times, 50)
        p95_time = np.percentile(inference_times, 95)
        p99_time = np.percentile(inference_times, 99)

        # Print results
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total frames processed: {args.num_frames}")
        logger.info(f"Total time:             {total_time:.3f} s")
        logger.info(f"Overall throughput:     {args.num_frames/total_time:.2f} FPS")
        logger.info("DPU Inference Statistics:")
        logger.info(f"  Mean:   {mean_time*1000:.3f} ms  ({1.0/mean_time:.2f} FPS)")
        logger.info(f"  Std:    {std_time*1000:.3f} ms")
        logger.info(f"  Min:    {min_time*1000:.3f} ms  ({1.0/min_time:.2f} FPS)")
        logger.info(f"  Max:    {max_time*1000:.3f} ms  ({1.0/max_time:.2f} FPS)")
        logger.info(f"  Median: {p50_time*1000:.3f} ms")
        logger.info(f"  P95:    {p95_time*1000:.3f} ms")
        logger.info(f"  P99:    {p99_time*1000:.3f} ms")
        logger.info("=" * 60)

    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()
