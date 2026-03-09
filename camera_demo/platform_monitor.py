#!/usr/bin/env python3
"""Platform statistics monitoring for Kria board."""

import re
import subprocess
import threading
import time


class PlatformMonitor:
    """Monitor platform statistics from xmutil xlnx_platformstats."""

    def __init__(self, update_interval=2.0):
        """Initialize the platform monitor.

        Args:
            update_interval: Interval in seconds to update platform stats (default: 2.0)
        """
        self.stats = {}
        self.stats_lock = threading.Lock()
        self.update_interval = update_interval
        self.monitor_thread = None
        self.running = False

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

    def update_stats(self):
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
                with self.stats_lock:
                    self.stats = stats
            else:
                print(f"Error running xmutil: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("xmutil command timed out")
        except Exception as e:
            print(f"Error updating platform stats: {e}")

    def monitor_loop(self):
        """Background thread loop to periodically update platform stats."""
        print(f"Starting platform stats monitor (interval: {self.update_interval}s)")
        while self.running:
            self.update_stats()
            time.sleep(self.update_interval)

    def start(self):
        """Start the platform stats monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self.monitor_loop,
                daemon=True
            )
            self.monitor_thread.start()

    def stop(self):
        """Stop the platform stats monitoring thread."""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)

    def get_stats(self):
        """Get current platform stats in a thread-safe manner.

        Returns:
            dict: Copy of current platform statistics
        """
        with self.stats_lock:
            return self.stats.copy()
