

import unittest
from unittest.mock import patch, Mock
import os
import sys
import subprocess
import requests

# Add the .autodev directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_provider import LLMProvider

class TestLLMProvider(unittest.TestCase):

    def setUp(self):
        self.provider = LLMProvider()

    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_plan_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='{"backlog":[]}', stderr='', returncode=0)
        
        result = self.provider.plan("test prompt")
        self.assertEqual(result, {"backlog": []})
        mock_subprocess_run.assert_called_once()

    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('llm_provider.LLMProvider._try_gemini')
    @patch('requests.post')
    def test_plan_gemini_fail_lmstudio_success(self, mock_requests_post, mock_try_gemini, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        
        # Simulate Gemini failing and LM Studio succeeding
        mock_try_gemini.return_value = (False, None)
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": '{"backlog":[]}'}}]},
            raise_for_status=lambda: None
        )
        
        result = self.provider.plan("test prompt")
        self.assertEqual(result, {"backlog": []})
        mock_requests_post.assert_called_once()
    
    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_code_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='print("Hello, World!")', stderr='', returncode=0)
        
        result = self.provider.code("test prompt")
        self.assertEqual(result, 'print("Hello, World!")')
        mock_subprocess_run.assert_called_once()

    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('llm_provider.LLMProvider._try_gemini')
    @patch('requests.post')
    def test_code_gemini_fail_lmstudio_success(self, mock_requests_post, mock_try_gemini, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        
        # Simulate Gemini failing and LM Studio succeeding
        mock_try_gemini.return_value = (False, None)
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": 'print("Hello, World!")'}}]},
            raise_for_status=lambda: None
        )
        
        result = self.provider.code("test prompt")
        self.assertEqual(result, 'print("Hello, World!")')
        mock_requests_post.assert_called_once()

    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    def test_troubleshoot_gemini_success(self, mock_subprocess_run, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'
        mock_subprocess_run.return_value = Mock(stdout='reboot the server', stderr='', returncode=0)

        result = self.provider.troubleshoot("test prompt")
        self.assertEqual(result, 'reboot the server')
        mock_subprocess_run.assert_called_once()

    @patch('llm_provider.LLMProvider._find_gemini_cli')
    @patch('subprocess.run')
    @patch('requests.post')
    def test_troubleshoot_gemini_fail_lmstudio_success(self, mock_requests_post, mock_subprocess_run, mock_find_gemini_cli):
        mock_find_gemini_cli.return_value = '/fake/gemini'

        # Simulate Gemini failing and LM Studio succeeding
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "Command failed")
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": 'reboot the server'}}]},
            raise_for_status=lambda: None
        )

        result = self.provider.troubleshoot("test prompt")
        self.assertEqual(result, 'reboot the server')
        mock_requests_post.assert_called_once()

    @patch('requests.post')
    def test_lmstudio_trims_long_prompt(self, mock_requests_post):
        provider = LLMProvider()
        provider.lmstudio_context_limit = 512  # force aggressive trimming for the test
        provider._select_model_for_lmstudio = lambda task_type: "rnj-1"  # type: ignore
        mock_requests_post.return_value = Mock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            raise_for_status=lambda: None,
        )

        long_prompt = "x" * 5000
        provider._try_lmstudio(long_prompt, "code")
        sent_prompt = mock_requests_post.call_args[1]["json"]["messages"][0]["content"]
        self.assertLess(len(sent_prompt), len(long_prompt))
        self.assertIn("TRUNCATED", sent_prompt)

    @patch('requests.post')
    def test_lmstudio_retries_on_context_error(self, mock_requests_post):
        provider = LLMProvider()
        provider._select_model_for_lmstudio = lambda task_type: "rnj-1"  # type: ignore

        context_error = requests.exceptions.HTTPError("context length of only 4096 tokens")
        success = Mock(
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
            raise_for_status=lambda: None,
        )
        mock_requests_post.side_effect = [context_error, success]

        result = provider._try_lmstudio("prompt", "plan")
        self.assertEqual(result, "ok")
        self.assertEqual(mock_requests_post.call_count, 2)

if __name__ == '__main__':
    unittest.main()
