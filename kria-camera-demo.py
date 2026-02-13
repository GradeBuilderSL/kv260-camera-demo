#!/usr/bin/env python3
"""Kria Camera Demo - Entry point for RealSense Camera with DPU Inference."""

import cv2
import time
import argparse
from pathlib import Path

from camera_demo import (
    check_and_elevate_privileges,
    KriaCameraDemo,
    get_contrasting_color,
    draw_text_with_background,
    load_model_metadata
)


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Kria Camera Demo - RealSense camera with DPU inference",
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

    return parser.parse_args()


def main():
    """Main application loop."""
    # Parse command-line arguments
    args = parse_arguments()

    # Check for root privileges and elevate if needed
    check_and_elevate_privileges()

    # Load model metadata from meta.json
    print(f"Loading model metadata from {args.model_dir}/meta.json...")
    model_info = load_model_metadata(args.model_dir)

    print(f"Model file: {model_info['model_path']}")
    print(f"Prototxt file: {model_info['prototxt_path']}")
    print(f"Target: {model_info['metadata'].get('target', 'Unknown')}")

    # Initialize the camera demo
    demo = KriaCameraDemo(
        prototxt_path=model_info['prototxt_path'],
        model_path=model_info['model_path'],
        dpu_bit_path=args.dpu_bit,
        labels_path=args.labels
    )

    try:
        demo.start_camera()
        demo.start_platform_monitor()

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
            perf_text = f"Display FPS: {actual_fps:.1f} | DPU Delay: {inference_time*1000:.1f}ms ({dpu_fps:.1f} FPS)"
            perf_font = cv2.FONT_HERSHEY_SIMPLEX
            perf_scale = 0.6
            perf_thickness = 2
            perf_pos = (10, 20)

            # Get text size for contrast calculation
            (perf_width, perf_height), _ = cv2.getTextSize(
                perf_text, perf_font, perf_scale, perf_thickness
            )

            # Determine contrasting color based on background
            perf_color = get_contrasting_color(
                display_image, perf_pos[0], perf_pos[1] - perf_height,
                perf_width, perf_height
            )

            # Draw performance text with semi-transparent background
            draw_text_with_background(
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
            header_color = get_contrasting_color(
                display_image, header_pos[0], header_pos[1] - header_height,
                header_width, header_height
            )

            # Draw section header with semi-transparent background
            draw_text_with_background(
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
                text_color = get_contrasting_color(
                    display_image, pred_pos[0], pred_pos[1] - text_height,
                    text_width, text_height
                )

                # Draw prediction text with semi-transparent background
                draw_text_with_background(
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
                header_color = get_contrasting_color(
                    display_image, header_pos[0], header_pos[1] - header_height,
                    header_width, header_height
                )

                # Draw section header with semi-transparent background
                draw_text_with_background(
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
                    text_color = get_contrasting_color(
                        display_image, stats_pos[0], stats_pos[1] - text_height,
                        text_width, text_height
                    )

                    # Draw stats text with semi-transparent background
                    draw_text_with_background(
                        display_image, stats_text, stats_pos, stats_font,
                        stats_scale, text_color, stats_thickness, bg_opacity=0.5
                    )

                    y_offset += 20

            # Add "Press 'q' to exit" notification at the bottom of the frame
            img_height = display_image.shape[0]
            exit_text = "Press 'q' to exit"
            exit_font = cv2.FONT_HERSHEY_SIMPLEX
            exit_scale = 0.6
            exit_thickness = 2

            # Get text size to position at bottom-right
            (exit_width, exit_height), _ = cv2.getTextSize(
                exit_text, exit_font, exit_scale, exit_thickness
            )

            exit_pos = (display_image.shape[1] - exit_width - 10, img_height - 10)

            # Determine contrasting color based on background
            exit_color = get_contrasting_color(
                display_image, exit_pos[0], exit_pos[1] - exit_height,
                exit_width, exit_height
            )

            # Draw exit instruction with semi-transparent background
            draw_text_with_background(
                display_image, exit_text, exit_pos, exit_font,
                exit_scale, exit_color, exit_thickness, bg_opacity=0.5
            )

            # Show image
            cv2.imshow('RealSense - Color', display_image)

            # Break loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
