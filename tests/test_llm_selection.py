import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import logging
from pathlib import Path





from llm_provider import LLMProvider

# Configure logging to capture INFO messages during tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestLLMProviderModelSelection(unittest.TestCase):

    def setUp(self):
        # Ensure lm_studio_model_benchmarks.json exists for the tests
        self.benchmark_file_path = Path(__file__).resolve().parent.parent / "lm_studio_model_benchmarks.json"
        if not self.benchmark_file_path.exists():
            # Recreate the file if it was deleted
            with open(self.benchmark_file_path, "w", encoding="utf-8") as f:
                f.write("{\n    \"Hermes 2 Pro 7B\": {\n        \"vram_gb\": 5,\n        \"ram_gb\": 8,\n        \"performance_score\": 0.8,\n        \"context_window\": 4096\n    },\n    \"Mistral 7B\": {\n        \"vram_gb\": 4,\n        \"ram_gb\": 7,\n        \"performance_score\": 0.75,\n        \"context_window\": 8192\n    },\n    \"Nous-Hermes 2 7B\": {\n        \"vram_gb\": 5,\n        \"ram_gb\": 8,\n        \"performance_score\": 0.82,\n        \"context_window\": 4096\n    },\n    \"Qwen2.5 Coder 7B\": {\n        \"vram_gb\": 6,\n        \"ram_gb\": 10,\n        \"performance_score\": 0.85,\n        \"context_window\": 8192\n    },\n    \"StarCoder 7B\": {\n        \"vram_gb\": 6,\n        \"ram_gb\": 10,\n        \"performance_score\": 0.8,\n        \"context_window\": 8192\n    },\n    \"CodeGemma 7B\": {\n        \"vram_gb\": 5,\n        \"ram_gb\": 9,\n        \"performance_score\": 0.83,\n        \"context_window\": 8192\n    },\n    \"MixtralInstruct 8x7B\": {\n        \"vram_gb\": 12,\n        \"ram_gb\": 24,\n        \"performance_score\": 0.9,\n        \"context_window\": 32768\n    },\n    \"rnj-1\": {\n        \"vram_gb\": 8,\n        \"ram_gb\": 16,\n        \"performance_score\": 0.88,\n        \"context_window\": 4096\n    },\n    \"TinyLlama-1.1B\": {\n        \"vram_gb\": 1,\n        \"ram_gb\": 2,\n        \"performance_score\": 0.5,\n        \"context_window\": 2048\n    }\n}")

        # Clear any potential environment variables that might interfere with tests
        for key in ["LMSTUDIO_MODEL", "LMSTUDIO_MODEL_PLAN", "LMSTUDIO_MODEL_CODE", "LMSTUDIO_MODEL_TROUBLESHOOT"]:
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        # Clean up environment variables
        for key in ["LMSTUDIO_MODEL", "LMSTUDIO_MODEL_PLAN", "LMSTUDIO_MODEL_CODE", "LMSTUDIO_MODEL_TROUBLESHOOT"]:
            if key in os.environ:
                del os.environ[key]

    @patch('llm_provider.psutil')
    @patch('llm_provider.subprocess')

    def test_low_resource_selection(self, mock_subprocess, mock_psutil, mocker):
        """Test selection when only a small model fits."""
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value = MagicMock(total=4 * (1024**3)) # 4GB RAM
        
        # Mock GPU VRAM detection to return 2GB
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="2048 MiB\n", stderr="" # For nvidia-smi
        )
        mock_available_models.return_value = [
            "Hermes 2 Pro 7B", "Mistral 7B", "TinyLlama-1.1B", "MixtralInstruct 8x7B"
        ]

        provider = llm_provider.LLMProvider(max_vram_gb=2) # Explicitly set a low max_vram_gb
        mock_available_models_list = [
            "Hermes 2 Pro 7B", "Mistral 7B", "TinyLlama-1.1B", "MixtralInstruct 8x7B"
        ]
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("code")
        self.assertEqual(selected, "TinyLlama-1.1B")
        self.assertLessEqual(provider.max_vram_gb, 2) # Ensure detected VRAM is respected
        self.assertLessEqual(provider.total_ram_gb, 4)

    @patch('llm_provider.psutil')
    @patch('llm_provider.subprocess')

    def test_high_resource_selection(self, mock_subprocess, mock_psutil, mocker):
        """Test selection when larger models are available and fit."""
        mock_psutil.cpu_count.return_value = 16
        mock_psutil.virtual_memory.return_value = MagicMock(total=32 * (1024**3)) # 32GB RAM

        # Mock GPU VRAM detection to return 16GB
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="16384 MiB\n", stderr="" # For nvidia-smi
        )
        mock_available_models.return_value = [
            "Hermes 2 Pro 7B", "Mistral 7B", "TinyLlama-1.1B", "MixtralInstruct 8x7B",
            "Qwen2.5 Coder 7B", "rnj-1"
        ]

        provider = LLMProvider(max_vram_gb=16)
        mock_available_models_list = [
            "Hermes 2 Pro 7B", "Mistral 7B", "TinyLlama-1.1B", "MixtralInstruct 8x7B",
            "Qwen2.5 Coder 7B", "rnj-1"
        ]
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("troubleshoot")
        # MixtralInstruct 8x7B (VRAM: 12, RAM: 24, Perf: 0.9) should be selected for troubleshoot
        self.assertEqual(selected, "MixtralInstruct 8x7B")
        self.assertLessEqual(provider.max_vram_gb, 16)
        self.assertLessEqual(provider.total_ram_gb, 32)
        
    @patch('llm_provider.psutil')
    @patch('llm_provider.subprocess')

    def test_env_override_selection(self, mock_subprocess, mock_psutil, mocker):
        """Test that environment variable overrides are respected."""
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024**3)) # 16GB RAM
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="8192 MiB\n", stderr="" # 8GB VRAM
        )
        mock_available_models.return_value = [
            "Hermes 2 Pro 7B", "TinyLlama-1.1B", "rnj-1", "CodeGemma 7B"
        ]

        mock_available_models_list = [
            "Hermes 2 Pro 7B", "TinyLlama-1.1B", "rnj-1", "CodeGemma 7B"
        ]

        os.environ["LMSTUDIO_MODEL_CODE"] = "CodeGemma 7B"
        provider = LLMProvider()
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("code")
        self.assertEqual(selected, "CodeGemma 7B")

        # Test global override
        del os.environ["LMSTUDIO_MODEL_CODE"]
        os.environ["LMSTUDIO_MODEL"] = "TinyLlama-1.1B"
        provider = LLMProvider() # Re-init to pick up new env var
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("plan")
        self.assertEqual(selected, "TinyLlama-1.1B")
        
        # Test override for non-existent model
        os.environ["LMSTUDIO_MODEL_PLAN"] = "NonExistentModel"
        provider = LLMProvider()
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("plan")
        # It should fallback to the best available model that fits if override is not found
        # In this case, rnj-1 (VRAM 8, RAM 16, Perf 0.88) would fit
        self.assertEqual(selected, "rnj-1")

    @patch('llm_provider.psutil')
    @patch('llm_provider.subprocess')

    def test_no_model_fits(self, mock_subprocess, mock_psutil, mocker):
        """Test scenario where no available model fits hardware constraints."""
        mock_psutil.cpu_count.return_value = 2
        mock_psutil.virtual_memory.return_value = MagicMock(total=1 * (1024**3)) # 1GB RAM
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="512 MiB\n", stderr="" # 0.5GB VRAM
        )
        mock_available_models.return_value = [
            "Hermes 2 Pro 7B", "Mistral 7B", "MixtralInstruct 8x7B"
        ] # All are too large

        provider = LLMProvider(max_vram_gb=0) # Effectively no VRAM
        mock_available_models_list = [
            "Hermes 2 Pro 7B", "Mistral 7B", "MixtralInstruct 8x7B"
        ] # All are too large
        mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
        selected = provider._select_model_for_lmstudio("code")
        self.assertIsNone(selected)
        
    @patch('llm_provider.psutil')
    @patch('llm_provider.subprocess')

    def test_incomplete_benchmark_data(self, mock_subprocess, mock_psutil, mocker):
        """Test that models with incomplete benchmark data are skipped."""
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * (1024**3)) # 16GB RAM
        mock_subprocess.run.return_value = MagicMock(
            returncode=0, stdout="8192 MiB\n", stderr="" # 8GB VRAM
        )
        mock_available_models_list = [
            "rnj-1", "ModelWithNoVRAMInfo", "TinyLlama-1.1B"
        ]
        
        # Temporarily modify benchmark data to simulate incomplete info
        original_benchmarks = llm_provider.LLMProvider._get_model_info_from_benchmarks(MagicMock()) # Get existing instance for modifying
        modified_benchmarks = original_benchmarks.copy()
        modified_benchmarks["ModelWithNoVRAMInfo"] = {"ram_gb": 4, "performance_score": 0.6} # Missing vram_gb
        
        with patch('llm_provider.LLMProvider._get_model_info_from_benchmarks', return_value=modified_benchmarks):
            provider = llm_provider.LLMProvider()
            mocker.patch.object(provider, '_lmstudio_available_models', return_value=mock_available_models_list)
            selected = provider._select_model_for_lmstudio("plan")
            # rnj-1 fits, TinyLlama-1.1B fits. rnj-1 has better performance.
            self.assertEqual(selected, "rnj-1")


if __name__ == '__main__':
    unittest.main()
