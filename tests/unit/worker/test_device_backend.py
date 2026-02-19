"""Tests defining GPU backend detection."""

import pytest
from unittest.mock import patch, MagicMock


class TestDeviceBackend:
    """Tests for detect_backend() function."""

    def test_detect_mps_when_available(self):
        """Detect MPS on Apple Silicon."""
        with patch("worker.models.llm.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = True

            from worker.models.llm import detect_backend

            assert detect_backend() == "mps"

    def test_detect_cuda_when_mps_unavailable(self):
        """Detect CUDA when MPS unavailable but CUDA is."""
        with patch("worker.models.llm.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = True

            from worker.models.llm import detect_backend

            assert detect_backend() == "cuda"

    def test_fallback_to_cpu(self):
        """Fall back to CPU when no GPU available."""
        with patch("worker.models.llm.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False

            from worker.models.llm import detect_backend

            assert detect_backend() == "cpu"

    def test_mps_prioritized_over_cuda(self):
        """MPS is checked before CUDA (shouldn't happen, but verify priority)."""
        with patch("worker.models.llm.torch") as mock_torch:
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = True

            from worker.models.llm import detect_backend

            # MPS should be returned even if CUDA is available
            assert detect_backend() == "mps"
