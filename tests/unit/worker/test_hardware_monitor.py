"""Tests defining HardwareMonitor async context manager API."""

import asyncio
import pytest
from unittest.mock import Mock, patch


class TestHardwareMonitor:
    """Tests for HardwareMonitor async context manager."""

    @pytest.mark.asyncio
    async def test_captures_max_ram(self):
        """max_ram_usage_mb = peak RAM during context."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            # Samples: entry, loop samples, exit
            mock_psutil.virtual_memory.side_effect = [
                Mock(used=500 * 1024 * 1024),  # 500 MB (entry sample)
                Mock(used=800 * 1024 * 1024),  # 800 MB (peak, loop)
                Mock(used=700 * 1024 * 1024),  # 700 MB (loop)
                Mock(used=600 * 1024 * 1024),  # 600 MB (loop)
                Mock(used=650 * 1024 * 1024),  # 650 MB (exit sample)
            ]
            # First call is prime (ignored), rest are samples
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)
            mock_psutil.sensors_temperatures.return_value = {}

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.2)

            assert monitor.metrics.max_ram_usage_mb == pytest.approx(800, rel=0.01)

    @pytest.mark.asyncio
    async def test_calculates_avg_cpu(self):
        """avg_cpu_utilization_pct = mean of samples."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            # First call is prime (ignored), rest are samples: entry, loop, loop, loop, exit
            mock_psutil.cpu_percent.side_effect = [0.0, 40.0, 60.0, 80.0, 70.0, 50.0]
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)
            mock_psutil.sensors_temperatures.return_value = {}

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.2)

            # Mean of samples (excluding the 0.0 prime call)
            assert monitor.metrics.avg_cpu_utilization_pct == pytest.approx(60.0, abs=15)

    @pytest.mark.asyncio
    async def test_handles_missing_temperature(self):
        """peak_cpu_temp_c = None when no sensors."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            mock_psutil.sensors_temperatures.return_value = {}  # No sensors
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.1)

            assert monitor.metrics.peak_cpu_temp_c is None

    @pytest.mark.asyncio
    async def test_tracks_swap_delta(self):
        """swap_in/out_bytes = delta from start to end."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            mock_psutil.swap_memory.side_effect = [
                Mock(sin=1000, sout=500),  # Start
                Mock(sin=1500, sout=800),  # End
            ]
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.sensors_temperatures.return_value = {}

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.1)

            # Delta: sin=1500-1000=500, sout=800-500=300
            assert monitor.metrics.swap_in_bytes == 500
            assert monitor.metrics.swap_out_bytes == 300

    @pytest.mark.asyncio
    async def test_captures_peak_temperature(self):
        """peak_cpu_temp_c = max temperature observed."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            # Samples: entry, loop, loop, loop, exit
            mock_psutil.sensors_temperatures.side_effect = [
                {"coretemp": [Mock(current=65.0)]},  # entry
                {"coretemp": [Mock(current=72.0)]},  # Peak (loop)
                {"coretemp": [Mock(current=68.0)]},  # loop
                {"coretemp": [Mock(current=70.0)]},  # loop
                {"coretemp": [Mock(current=66.0)]},  # exit
            ]
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.2)

            assert monitor.metrics.peak_cpu_temp_c == 72.0

    @pytest.mark.asyncio
    async def test_captures_samples_for_fast_operations(self):
        """Should capture samples even when operation completes quickly."""
        with patch("worker.services.hardware_monitor.psutil") as mock_psutil:
            # First call is prime (0.0, ignored), entry sample, exit sample
            mock_psutil.cpu_percent.side_effect = [0.0, 50.0, 60.0]
            mock_psutil.virtual_memory.side_effect = [
                Mock(used=500 * 1024 * 1024),  # entry sample
                Mock(used=600 * 1024 * 1024),  # exit sample
            ]
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)
            mock_psutil.sensors_temperatures.return_value = {}

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=1.0) as monitor:
                # Operation completes immediately - no time for loop samples
                pass

            # Should have at least 2 samples (entry + exit)
            # avg_cpu should NOT be 0.0 (the primed value)
            assert monitor.metrics.avg_cpu_utilization_pct == pytest.approx(55.0, abs=1)
            assert monitor.metrics.max_ram_usage_mb == pytest.approx(600, rel=0.01)
