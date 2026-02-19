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
            mock_psutil.virtual_memory.side_effect = [
                Mock(used=500 * 1024 * 1024),  # 500 MB
                Mock(used=800 * 1024 * 1024),  # 800 MB (peak)
                Mock(used=600 * 1024 * 1024),  # 600 MB
            ]
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
            mock_psutil.cpu_percent.side_effect = [40.0, 60.0, 80.0]
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)
            mock_psutil.sensors_temperatures.return_value = {}

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.2)

            # Mean of [40, 60, 80] = 60
            assert monitor.metrics.avg_cpu_utilization_pct == pytest.approx(60.0, abs=10)

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
            mock_psutil.sensors_temperatures.side_effect = [
                {"coretemp": [Mock(current=65.0)]},
                {"coretemp": [Mock(current=72.0)]},  # Peak
                {"coretemp": [Mock(current=68.0)]},
            ]
            mock_psutil.virtual_memory.return_value = Mock(used=100 * 1024 * 1024)
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.swap_memory.return_value = Mock(sin=0, sout=0)

            from worker.services.hardware_monitor import HardwareMonitor

            async with HardwareMonitor(sample_interval=0.05) as monitor:
                await asyncio.sleep(0.2)

            assert monitor.metrics.peak_cpu_temp_c == 72.0
