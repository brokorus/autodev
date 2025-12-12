import unittest
from unittest.mock import patch, Mock
import os
import sys
import subprocess
from types import SimpleNamespace
import requests

# Add the .autodev directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_provider import LLMProvider


class TestLLMProvider(unittest.TestCase):

    def setUp(self):
        self.provider = LLMProvider()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_plan_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='{"backlog":[]}', stderr='', returncode=0)
        mock_get.return_value = Mock(json=lambda: {"data": []}, raise_for_status=lambda: None)

        result = self.provider.plan("test prompt")
        self.assertEqual(result, {"backlog": []})
        mock_subprocess_run.assert_called_once()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('llm_provider.LLMProvider._try_gemini')
    @patch('requests.post')
    @patch('llm_provider.LLMProvider._lmstudio_prepare_model')
    def test_plan_gemini_fail_lmstudio_success(self, mock_prepare, mock_requests_post, mock_try_gemini, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_prepare.return_value = True

        # Simulate Gemini failing and LM Studio succeeding
        mock_try_gemini.return_value = (False, None)
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": '{"backlog":[]}'}}]},
            raise_for_status=lambda: None
        )
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "rnj-1"}]}, raise_for_status=lambda: None)

        result = self.provider.plan("test prompt")
        self.assertEqual(result, {"backlog": []})
        mock_requests_post.assert_called_once()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_code_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='print("Hello, World!")', stderr='', returncode=0)
        mock_get.return_value = Mock(json=lambda: {"data": []}, raise_for_status=lambda: None)

        result = self.provider.code("test prompt")
        self.assertEqual(result, 'print("Hello, World!")')
        mock_subprocess_run.assert_called_once()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('llm_provider.LLMProvider._try_gemini')
    @patch('requests.post')
    @patch('llm_provider.LLMProvider._lmstudio_prepare_model')
    def test_code_gemini_fail_lmstudio_success(self, mock_prepare, mock_requests_post, mock_try_gemini, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_prepare.return_value = True

        # Simulate Gemini failing and LM Studio succeeding
        mock_try_gemini.return_value = (False, None)
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": 'print("Hello, World!")'}}]},
            raise_for_status=lambda: None
        )
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "rnj-1"}]}, raise_for_status=lambda: None)

        result = self.provider.code("test prompt")
        self.assertEqual(result, 'print("Hello, World!")')
        mock_requests_post.assert_called_once()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_troubleshoot_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='reboot the server', stderr='', returncode=0)
        mock_get.return_value = Mock(json=lambda: {"data": []}, raise_for_status=lambda: None)

        result = self.provider.troubleshoot("test prompt")
        self.assertEqual(result, 'reboot the server')
        mock_subprocess_run.assert_called_once()

    @patch('requests.get')
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    @patch('requests.post')
    @patch('llm_provider.LLMProvider._lmstudio_prepare_model')
    def test_troubleshoot_gemini_fail_lmstudio_success(self, mock_prepare, mock_requests_post, mock_subprocess_run, mock_find_gemini_cli, mock_get):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_prepare.return_value = True

        # Simulate Gemini failing and LM Studio succeeding
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "Command failed")
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": 'reboot the server'}}]},
            raise_for_status=lambda: None,
        )
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "rnj-1"}]}, raise_for_status=lambda: None)

        result = self.provider.troubleshoot("test prompt")
        self.assertEqual(result, 'reboot the server')
        mock_requests_post.assert_called_once()

    @patch('requests.get')
    @patch('requests.post')
    @patch('llm_provider.LLMProvider._lmstudio_prepare_model')
    def test_lmstudio_trims_long_prompt(self, mock_prepare, mock_requests_post, mock_get):
        provider = LLMProvider()
        provider.lmstudio_context_limit = 512  # force aggressive trimming for the test
        provider._select_model_for_lmstudio = lambda task_type, available=None, estimate_ctx_tokens=None: "rnj-1"  # type: ignore
        mock_prepare.return_value = True
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            raise_for_status=lambda: None,
        )
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "rnj-1"}]}, raise_for_status=lambda: None)

        long_prompt = "x" * 5000
        provider._try_lmstudio(long_prompt, "code")
        sent_prompt = mock_requests_post.call_args[1]["json"]["messages"][0]["content"]
        self.assertLess(len(sent_prompt), len(long_prompt))
        self.assertIn("TRUNCATED", sent_prompt)

    @patch('requests.get')
    @patch('requests.post')
    @patch('llm_provider.LLMProvider._lmstudio_prepare_model')
    def test_lmstudio_retries_on_context_error(self, mock_prepare, mock_requests_post, mock_get):
        provider = LLMProvider()
        mock_prepare.return_value = True
        provider._select_model_for_lmstudio = lambda task_type, available=None, estimate_ctx_tokens=None: "rnj-1"  # type: ignore

        context_error = requests.exceptions.HTTPError("context length of only 4096 tokens")
        success = Mock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            raise_for_status=lambda: None,
        )
        mock_requests_post.side_effect = [context_error, success]
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "rnj-1"}]}, raise_for_status=lambda: None)

        result = provider._try_lmstudio("prompt", "plan")
        self.assertEqual(result, "ok")
        self.assertEqual(mock_requests_post.call_count, 2)

    @patch('requests.get')
    def test_lmstudio_model_override_preferred(self, mock_get):
        provider = LLMProvider()
        os.environ["LMSTUDIO_MODEL_CODE"] = "custom-model"
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "custom-model"}]}, raise_for_status=lambda: None)
        model = provider._select_model_for_lmstudio("code", ["custom-model"])
        self.assertEqual(model, "custom-model")
        os.environ.pop("LMSTUDIO_MODEL_CODE", None)

    @patch('requests.get')
    def test_lmstudio_falls_back_to_available_model(self, mock_get):
        provider = LLMProvider()
        provider.max_vram_gb = 16
        mock_get.return_value = Mock(json=lambda: {"data": [{"id": "some-model-7B"}]}, raise_for_status=lambda: None)
        model = provider._select_model_for_lmstudio("plan", ["some-model-7B"])
        self.assertIsNotNone(model)

    @patch('llm_provider.LLMProvider._lmstudio_unload_all_models')
    @patch('llm_provider.subprocess.run')
    @patch('llm_provider.platform.system', return_value="Windows")
    def test_detect_gpu_vram_prefers_largest_adapter(self, mock_platform, mock_run, mock_unload):
        mock_unload.return_value = None
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout="""AdapterRAM
2147483648
8589934592
0
"""
        )
        provider = LLMProvider(max_vram_gb=8)
        detected = provider._detect_gpu_vram_gb()
        self.assertEqual(detected, 8)  # picks 8GB over 2GB/0GB
        # max_vram_gb should also reflect the detected value (not the under-reported one)
        self.assertGreaterEqual(provider.max_vram_gb, 8)


if __name__ == '__main__':
    unittest.main()
