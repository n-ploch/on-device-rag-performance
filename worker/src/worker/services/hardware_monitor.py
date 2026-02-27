"""Async hardware monitor used around a single generation request."""

from __future__ import annotations

import asyncio
import logging
from statistics import mean

import psutil

logger = logging.getLogger(__name__)

from shared_types.schemas import HardwareMeasurement


class HardwareMonitor:
    """Collects lightweight RAM/CPU/temp samples during an async context."""

    def __init__(self, sample_interval: float = 0.2):
        self.sample_interval = sample_interval
        self.metrics = HardwareMeasurement(
            max_ram_usage_mb=0.0,
            avg_cpu_utilization_pct=0.0,
            peak_cpu_temp_c=None,
            swap_in_bytes=0,
            swap_out_bytes=0,
        )

        self._active = False
        self._task: asyncio.Task | None = None
        self._ram_samples_mb: list[float] = []
        self._cpu_samples_pct: list[float] = []
        self._temp_samples_c: list[float] = []
        self._swap_start = None

    async def __aenter__(self) -> "HardwareMonitor":
        self._swap_start = psutil.swap_memory()

        # Prime cpu_percent() - first call always returns 0.0
        psutil.cpu_percent()

        self._active = True

        # Capture first sample immediately before starting the loop
        self._capture_sample()

        self._task = asyncio.create_task(self._sample_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._active = False
        if self._task is not None:
            await self._task

        # Capture final sample to ensure we have measurements at exit
        try:
            self._capture_sample()
        except Exception as e:
            logger.warning("Final hardware sample error: %s", e)

        swap_end = psutil.swap_memory()
        swap_in = max(0, int(swap_end.sin - self._swap_start.sin))
        swap_out = max(0, int(swap_end.sout - self._swap_start.sout))

        self.metrics = HardwareMeasurement(
            max_ram_usage_mb=max(self._ram_samples_mb, default=0.0),
            avg_cpu_utilization_pct=mean(self._cpu_samples_pct) if self._cpu_samples_pct else 0.0,
            peak_cpu_temp_c=max(self._temp_samples_c) if self._temp_samples_c else None,
            swap_in_bytes=swap_in,
            swap_out_bytes=swap_out,
        )
        logger.debug(
            "Hardware sampling complete: %d RAM samples, %d CPU samples",
            len(self._ram_samples_mb),
            len(self._cpu_samples_pct),
        )

    async def _sample_loop(self) -> None:
        while self._active:
            try:
                self._capture_sample()
            except Exception as e:
                logger.warning("Hardware sampling error: %s", e)
                break
            await asyncio.sleep(self.sample_interval)

    def _capture_sample(self) -> None:
        ram_mb = psutil.virtual_memory().used / (1024 * 1024)
        self._ram_samples_mb.append(ram_mb)

        cpu_pct = float(psutil.cpu_percent())
        self._cpu_samples_pct.append(cpu_pct)

        # Temperature monitoring is only available on Linux
        # psutil.sensors_temperatures() does not exist on macOS/Windows
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures() or {}
                current_temps: list[float] = []
                for entries in temps.values():
                    for entry in entries:
                        value = getattr(entry, "current", None)
                        if value is not None:
                            current_temps.append(float(value))

                if current_temps:
                    self._temp_samples_c.append(max(current_temps))
            except Exception as e:
                # Some systems may have the method but fail to read sensors
                logger.debug("Could not read temperature sensors: %s", e)
