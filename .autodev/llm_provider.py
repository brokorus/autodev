
import psutil
import shutil
import json
import logging
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMProvider:
    def __init__(self, max_vram_gb: int = 8):
        # Maximum VRAM in GB available for models. Used to filter suitable LM Studio models.
        self.max_vram_gb = max_vram_gb
        # Path to the Gemini CLI executable. Discovered dynamically.
        self.gemini_path = self._find_gemini_cli()
        # The root directory of the project, used for resolving paths to app resources.
        self.project_root = Path(__file__).resolve().parent.parent
        # Determine if AutoDev is running in offline mode based on environment variable.
        offline_flag = os.getenv("AUTODEV_OFFLINE_MODE", "").lower()
        self.offline_mode = offline_flag in {"1", "true", "yes"}
        # LM Studio configuration parameters, loaded from environment variables or default values.
        self.lmstudio_context_limit = int(os.getenv("LMSTUDIO_CONTEXT_WINDOW", "4096"))
        self.lmstudio_max_tokens = max(128, int(os.getenv("LMSTUDIO_MAX_TOKENS", "1024")))
        self.lmstudio_timeout_seconds = int(os.getenv("LMSTUDIO_TIMEOUT_SECONDS", "30"))
        self.lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234")
        self.lmstudio_health_timeout = int(os.getenv("LMSTUDIO_HEALTH_TIMEOUT", "5"))
        self.lmstudio_retries = max(1, int(os.getenv("LMSTUDIO_RETRIES", "2")))
        # Optional LM Studio model overrides, allowing specific models for different task types or a general override.
        self.lmstudio_model_overrides = {
            "plan": os.getenv("LMSTUDIO_MODEL_PLAN"),
            "code": os.getenv("LMSTUDIO_MODEL_CODE"),
            "troubleshoot": os.getenv("LMSTUDIO_MODEL_TROUBLESHOOT"),
            "any": os.getenv("LMSTUDIO_MODEL"),
        }
        # Cache for LM Studio's available models to avoid frequent API calls. Stores (timestamp, list_of_models).
        self._lmstudio_model_cache: Tuple[float, List[str]] = (0.0, [])

        # System hardware detection
        self.cpu_cores: int = 1
        self.total_ram_gb: int = max_vram_gb # Fallback for now
        self._detect_system_hardware()

    def plan(self, prompt: str) -> Optional[Dict]:
        if self.offline_mode:
            return self._rule_based_plan()

        for _ in range(3):
            response = self._try_provider(prompt, "plan")
            parsed = self._parse_json_content(response)
            if parsed is not None:
                return parsed
        return self._rule_based_plan()

    def code(self, prompt: str) -> Optional[str]:
        if self.offline_mode:
            return self._rule_based_code(prompt, "code")

        return self._try_provider(prompt, "code")

    def troubleshoot(self, prompt: str) -> Optional[str]:
        if self.offline_mode:
            return self._rule_based_code(prompt, "troubleshoot")

        return self._try_provider(prompt, "troubleshoot")

    def _try_provider(self, prompt: str, task_type: str) -> Optional[str]:
        # Try Gemini first
        if not self.gemini_path:
            self.gemini_path = self._find_gemini_cli()

        if self.gemini_path:
            success, response = self._try_gemini(prompt)
            if success:
                return response

        # Fallback to LM Studio
        return self._try_lmstudio(prompt, task_type)

    def _validate_json(self, response: str) -> bool:
        try:
            json.loads(response)
            return True
        except json.JSONDecodeError:
            return False

    def _parse_json_content(self, content: Optional[Union[str, Dict]]) -> Optional[Dict]:
        """
        Handle common LLM response shapes:
        - Raw JSON string
        - JSON wrapped in code fences
        - Object with a 'response' field containing JSON
        """
        if content is None:
            return None

        def parse_text(text: str) -> Optional[Dict]:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # remove leading language tag if present
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "response" in parsed and isinstance(parsed["response"], str):
                    inner = parse_text(parsed["response"])
                    if inner:
                        return inner
                return parsed
            except (json.JSONDecodeError, TypeError):
                pass

            # Heuristic: grab the first JSON object if extra text wraps it
            import re

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                candidate = match.group(0)
                opens = candidate.count("{")
                closes = candidate.count("}")
                if opens > closes:
                    candidate = candidate + ("}" * (opens - closes))
                open_brackets = candidate.count("[")
                close_brackets = candidate.count("]")
                if open_brackets > close_brackets:
                    candidate = candidate + ("]" * (open_brackets - close_brackets))
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Last attempt: balance braces/brackets on the entire string if it contains backlog
            if "backlog" in cleaned:
                candidate = cleaned
                open_brace = candidate.count("{")
                close_brace = candidate.count("}")
                if open_brace > close_brace:
                    candidate += "}" * (open_brace - close_brace)
                open_bracket = candidate.count("[")
                close_bracket = candidate.count("]")
                if open_bracket > close_bracket:
                    candidate += "]" * (open_bracket - close_bracket)
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    return None
            return None

        if isinstance(content, dict):
            if "backlog" in content or "halt_reason" in content:
                return content
            nested = content.get("response") or content.get("content")
            if isinstance(nested, str):
                return parse_text(nested)
            return None

        if isinstance(content, str):
            return parse_text(content)

        return None

    def _approx_token_count(self, text: str) -> int:
        """
        Very rough tokenizer approximation to keep LM Studio requests within the
        model context. Assumes ~4 characters per token which matches llama.cpp
        style models closely enough for safety trimming.
        """
        return max(1, len(text) // 4)

    def _trim_prompt_for_lmstudio(self, prompt: str, max_ctx_tokens: int) -> Tuple[str, bool]:
        """
        Trim the prompt to fit within the target context window with a safety margin
        for generation tokens. Returns the (possibly) trimmed prompt and a flag
        indicating whether trimming occurred.
        """
        if max_ctx_tokens <= 0:
            return prompt, False

        # Keep room for generation and system overhead
        target_tokens = max(256, max_ctx_tokens - 512)
        approx_tokens = self._approx_token_count(prompt)
        if approx_tokens <= target_tokens:
            return prompt, False

        max_chars = target_tokens * 4
        head = prompt[: max_chars // 2]
        tail = prompt[-(max_chars - len(head)) :]
        truncated = (
            f"{head}\n\n...[TRUNCATED to fit LM Studio context, kept ~{target_tokens} of "
            f"~{approx_tokens} estimated tokens]...\n\n{tail}"
        )
        return truncated, True

    def _looks_like_context_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            needle in msg
            for needle in [
                "context length",
                "n_ctx",
                "context the overflows",
                "token limit",
            ]
        )

    def _find_gemini_cli(self) -> Optional[str]:
        """
        Locates the Gemini CLI executable on the system.
        It searches the system's PATH environment variable for the 'gemini' executable,
        considering '.exe' or '.cmd' extensions on Windows.
        Returns the full path to the executable if found, otherwise None.
        """
        if platform.system() == "Windows":
            for path in os.environ["PATH"].split(os.pathsep):
                for exe in ["gemini.exe", "gemini.cmd"]:
                    exe_path = os.path.join(path, exe)
                    if os.path.exists(exe_path):
                        return exe_path
        elif platform.system() == "Darwin": # macOS detection
            for path in os.environ["PATH"].split(os.pathsep):
                exe_path = os.path.join(path, "gemini")
                if os.path.exists(exe_path):
                    return exe_path
        else:
            # For Linux and other Unix-like systems
            for path in os.environ["PATH"].split(os.pathsep):
                exe_path = os.path.join(path, "gemini")
                if os.path.exists(exe_path):
                    return exe_path
        return None


    def _try_gemini(self, prompt: str) -> (bool, Optional[str]):
        """
        Attempts to invoke the Gemini CLI with the given prompt.
        It executes the Gemini command, captures stdout/stderr, and returns the response if successful.
        Handles FileNotFoundError if the Gemini CLI is not found, and CalledProcessError if the CLI
        returns an error.
        """
        try:
            command = [self.gemini_path, "--output-format", "json"]
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout:
                return True, result.stdout
            return False, None
        except FileNotFoundError:
            logging.error("Gemini CLI not found.")
            return False, None
        except subprocess.CalledProcessError as e:
            logging.error(f"Gemini CLI failed with error: {e.stderr}")
            return False, None
        except Exception as e:
            logging.error(f"An unexpected error occurred with Gemini: {e}")
            return False, None

    def _try_lmstudio(self, prompt: str, task_type: str) -> Optional[str]:
        """
        Attempts to get a response from LM Studio. This function acts as a fallback
        if the Gemini CLI is not available or fails. It selects an appropriate model,
        trims the prompt if necessary to fit the context window, and retries the request
        if there are transient errors or context overflow issues.
        """
        available_models = self._lmstudio_available_models()
        model = self._select_model_for_lmstudio(task_type, available_models)
        if not model:
            # Log a warning if no suitable LM Studio model could be selected.
            logging.warning("No suitable LM Studio model found or selected.")
            return None

        timeout = self.lmstudio_timeout_seconds
        ctx_windows = [self.lmstudio_context_limit]
        # second pass with a smaller window for aggressive trimming if the first fails
        if self.lmstudio_context_limit > 1024:
            ctx_windows.append(max(1024, self.lmstudio_context_limit // 2))

        try:
            import requests

            url = f"{self.lmstudio_base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            last_error: Optional[Exception] = None
            for attempt in range(self.lmstudio_retries):
                for ctx in ctx_windows:
                    safe_prompt, trimmed = self._trim_prompt_for_lmstudio(prompt, ctx)
                    if trimmed:
                        logging.warning(
                            "LM Studio prompt trimmed to fit %s token context window (original ~%s tokens)",
                            ctx,
                            self._approx_token_count(prompt),
                        )
                    data = {
                        "model": model,
                        "messages": [{"role": "user", "content": safe_prompt}],
                        "max_tokens": self.lmstudio_max_tokens,
                        "temperature": 0.15,
                        "stream": False,
                    }
                    try:
                        response = requests.post(
                            url, headers=headers, json=data, timeout=timeout
                        )
                        response.raise_for_status()
                        payload = response.json()
                        return payload["choices"][0]["message"]["content"]
                    except requests.exceptions.RequestException as e:
                        last_error = e
                        # Retry once more with a smaller prompt if we hit a context issue
                        if self._looks_like_context_error(e) and ctx != ctx_windows[-1]:
                            logging.error(
                                "LM Studio reported context overflow; retrying with a smaller prompt"
                            )
                            continue
                        logging.error(
                            "LM Studio request failed (attempt %s/%s): %s",
                            attempt + 1,
                            self.lmstudio_retries,
                            e,
                        )
                        break
                    except (KeyError, ValueError, TypeError) as e:
                        last_error = e
                        logging.error(f"Failed to parse LM Studio response: {e}")
                        break

                if last_error:
                    # Backoff between attempts
                    time.sleep(min(2 * (attempt + 1), 5))

            if last_error:
                logging.error("LM Studio unavailable after retries: %s", last_error)
        except Exception as e:
            logging.error(f"An unexpected error occurred with LM Studio: {e}")
        return None

    def _lmstudio_available_models(self) -> List[str]:
        """
        Query LM Studio for available models. Caches results briefly to avoid
        spamming the local API.
        """
        now = time.time()
        cached_at, cached_models = self._lmstudio_model_cache
        if cached_models and (now - cached_at) < 30:
            return cached_models

        try:
            import requests

            url = f"{self.lmstudio_base_url.rstrip('/')}/v1/models"
            resp = requests.get(url, timeout=self.lmstudio_health_timeout)
            resp.raise_for_status()
            data = resp.json()
            models = [
                item.get("id")
                for item in data.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            self._lmstudio_model_cache = (now, models)
            return models
        except Exception as exc:
            logging.error(
                "LM Studio health check failed (%s). Is LM Studio running with the local server enabled?",
                exc,
            )
            return []

    def _required_vram_for_model(self, model: str) -> Optional[int]:
        """
        Estimates VRAM in GB for a model, prioritizing benchmark data, then heuristics.
        Returns None if no VRAM estimate can be made.
        """
        model_benchmarks = self._get_model_info_from_benchmarks()
        if model in model_benchmarks and "vram_gb" in model_benchmarks[model]:
            return model_benchmarks[model]["vram_gb"]

        # Fallback to heuristic if no benchmark data
        tokens = re.split(r"[^\w]+", model.lower())
        for token in tokens:
            if token.endswith("b") and token[:-1].isdigit():
                size = int(token[:-1])
                if size <= 7:
                    return 4
                if size == 8:
                    return 5
                if size == 9:
                    return 6
                if size >= 13:
                    return 10
        logging.warning(f"No VRAM benchmark data or heuristic match for model: {model}. Cannot determine VRAM requirements.")
        return None

    def _get_model_override(self, task_type: str) -> Optional[str]:
        # Re-read env on each call so runtime overrides are honored
        task_key = f"LMSTUDIO_MODEL_{task_type.upper()}"
        return os.getenv(task_key) or os.getenv("LMSTUDIO_MODEL")

    def _select_model_for_lmstudio(
        self, task_type: str, available_models: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Selects an appropriate LM Studio model based on the task type, available models,
        VRAM requirements, and environment variable overrides.
        Prioritizes:
        1. Task-specific environment variable override (e.g., LMSTUDIO_MODEL_PLAN).
        2. General environment variable override (LMSTUDIO_MODEL).
        3. Predefined model lists for each task type, filtered by available VRAM and presence
           in the currently available LM Studio models.
        4. Any available LM Studio model that fits the VRAM requirements as a last resort.
        """
        model_benchmarks = self._get_model_info_from_benchmarks()
        
        models_by_task = {
            "plan": ["rnj-1", "Hermes 2 Pro 7B", "Mistral 7B", "Nous-Hermes 2 7B", "TinyLlama-1.1B"],
            "code": ["rnj-1", "Qwen2.5 Coder 7B", "StarCoder 7B", "CodeGemma 7B", "TinyLlama-1.1B"],
            "troubleshoot": ["rnj-1", "MixtralInstruct 8x7B", "TinyLlama-1.1B"],
        }

        available_models = available_models or []
        model_benchmarks = self._get_model_info_from_benchmarks()

        # Task-specific or global override takes priority
        override = self._get_model_override(task_type)
        if override:
            if override not in available_models:
                logging.warning(
                    f"Requested LM Studio model override '{override}' for task '{task_type}' "
                    f"not found in available models: {available_models}. Falling back to defaults."
                )
            else:
                logging.info(f"Using LM Studio model override: '{override}' for task '{task_type}'.")
                return override

        eligible_models = []
        # First, consider models specifically suggested for the task type
        candidate_models = models_by_task.get(task_type, [])
        # Add models from other tasks and then filter by unique names.
        # This ensures that if a model is good for "code" and also available, it's considered for "plan" as well
        # if no specific "plan" model fits well.
        for other_task, models_list in models_by_task.items():
            if other_task != task_type:
                candidate_models.extend(models_list)
        
        # Ensure unique models and retain order based on task_type preference
        seen = set()
        unique_candidate_models = []
        for model_name in candidate_models:
            if model_name not in seen:
                unique_candidate_models.append(model_name)
                seen.add(model_name)
        
        # Iterate through unique candidate models first
        for model_name in unique_candidate_models:
            if model_name not in available_models:
                logging.debug(f"LM Studio model '{model_name}' not currently available in LM Studio.")
                continue

            model_info = model_benchmarks.get(model_name)
            if not model_info:
                logging.warning(f"No benchmark info found for model '{model_name}'. Skipping for intelligent selection.")
                continue

            vram_gb = model_info.get("vram_gb")
            ram_gb = model_info.get("ram_gb")
            performance_score = model_info.get("performance_score")

            if vram_gb is None or ram_gb is None or performance_score is None:
                logging.warning(f"Incomplete benchmark info for model '{model_name}'. Skipping for intelligent selection.")
                continue

            if vram_gb > self.max_vram_gb:
                logging.info(f"Model '{model_name}' (VRAM: {vram_gb} GB) exceeds system max VRAM ({self.max_vram_gb} GB).")
                continue
            if ram_gb > self.total_ram_gb:
                logging.info(f"Model '{model_name}' (RAM: {ram_gb} GB) exceeds system total RAM ({self.total_ram_gb} GB).")
                continue

            eligible_models.append((model_name, performance_score))
            logging.info(
                f"Model '{model_name}' is eligible (VRAM: {vram_gb}GB, RAM: {ram_gb}GB, Performance: {performance_score})."
            )

        if not eligible_models:
            logging.warning(
                "No eligible models found from predefined task lists that fit hardware constraints. "
                "Expanding search to all available LM Studio models."
            )
            # Fallback: Check all available models if none from the task-specific list fit
            for model_name in available_models:
                if (model_name, 0) in eligible_models: # Avoid re-adding if already processed
                    continue
                    
                model_info = model_benchmarks.get(model_name)
                if not model_info:
                    logging.warning(f"No benchmark info found for model '{model_name}'. Skipping for intelligent selection.")
                    continue

                vram_gb = model_info.get("vram_gb")
                ram_gb = model_info.get("ram_gb")
                performance_score = model_info.get("performance_score")

                if vram_gb is None or ram_gb is None or performance_score is None:
                    logging.warning(f"Incomplete benchmark info for model '{model_name}'. Skipping for intelligent selection.")
                    continue

                if vram_gb > self.max_vram_gb:
                    logging.info(f"Model '{model_name}' (VRAM: {vram_gb} GB) exceeds system max VRAM ({self.max_vram_gb} GB).")
                    continue
                if ram_gb > self.total_ram_gb:
                    logging.info(f"Model '{model_name}' (RAM: {ram_gb} GB) exceeds system total RAM ({self.total_ram_gb} GB).")
                    continue
                eligible_models.append((model_name, performance_score))
                logging.info(
                    f"Model '{model_name}' is eligible (VRAM: {vram_gb}GB, RAM: {ram_gb}GB, Performance: {performance_score})."
                )


        if eligible_models:
            # Sort by performance score (descending) and pick the best
            eligible_models.sort(key=lambda x: x[1], reverse=True)
            selected_model = eligible_models[0][0]
            logging.info(f"Selected LM Studio model: '{selected_model}' (Performance: {eligible_models[0][1]}).")
            return selected_model
        else:
            logging.error(
                "No suitable LM Studio model found that fits hardware constraints after checking "
                "predefined lists and all available models. "
                "Please check LM Studio is running and has models loaded, or adjust hardware."
            )
            return None

    # ---- Rule-based fallbacks -------------------------------------------------

    def _rule_based_plan(self) -> Dict:
        """
        Deterministic planner used when external LLMs are unavailable.
        Produces a single actionable backlog item if the dice roller feature
        has not been created yet.
        """
        app_dir = self.project_root / "crithappensdnd"
        dice_page = app_dir / "src" / "routes" / "dice" / "+page.svelte"
        backlog = []

        if not dice_page.exists():
            backlog.append(
                {
                    "idea": "Dice Roller",
                    "task": "Build a dice roller page at /dice with support for common dice, advantage/disadvantage, and a short roll history.",
                    "tests": [
                        "Playwright: user can roll dice, toggle advantage/disadvantage, and see history update"
                    ],
                    "priority": 1,
                    "prerequisites": {
                        "met": True,
                        "missing": [],
                        "blocked_due_to_paid_service": False,
                        "fifteen_factor_gates": [
                            "config-as-code",
                            "backing-services",
                            "logs-metrics",
                            "security",
                        ],
                        "infra": [],
                    },
                }
            )

        return {"halt_reason": "", "backlog": backlog}

    def _rule_based_code(self, prompt: str, task_type: str) -> str:
        """
        Deterministic coder/troubleshooter used when offline mode is enabled.
        """
        task_text = self._extract_task_from_prompt(prompt)
        lowered = task_text.lower()

        if "dice" in lowered and "roller" in lowered:
            return self._implement_dice_roller()

        if task_type == "troubleshoot":
            return "Offline troubleshoot mode: no deployment data available."

        return "Offline mode: no matching fallback for the requested task."

    def _detect_system_hardware(self) -> None:
        """Detects and sets system hardware properties (CPU, RAM, GPU VRAM)."""
        self.cpu_cores = self._detect_cpu_cores()
        self.total_ram_gb = self._detect_total_ram_gb()
        # self.max_vram_gb is already initialized, but we can try to improve it if possible
        # by detecting GPU VRAM dynamically.
        detected_vram = self._detect_gpu_vram_gb()
        if detected_vram is not None:
            self.max_vram_gb = min(self.max_vram_gb, detected_vram) # Use the lower of default/detected

        logging.info(f"Detected hardware: CPU Cores: {self.cpu_cores}, Total RAM: {self.total_ram_gb} GB, Max VRAM: {self.max_vram_gb} GB")

    def _detect_cpu_cores(self) -> int:
        """Detects the number of logical CPU cores."""
        try:
            return psutil.cpu_count(logical=True) or 1
        except Exception as e:
            logging.warning(f"Could not detect CPU cores: {e}. Defaulting to 1.")
            return 1

    def _detect_total_ram_gb(self) -> int:
        """Detects total system RAM in GB."""
        try:
            return int(psutil.virtual_memory().total / (1024**3))
        except Exception as e:
            logging.warning(f"Could not detect total RAM: {e}. Defaulting to {self.max_vram_gb} GB.")
            return self.max_vram_gb

    def _detect_gpu_vram_gb(self) -> Optional[int]:
        """
        Detects total GPU VRAM in GB using OS-specific commands.
        Returns None if detection fails or no suitable GPU is found.
        """
        system = platform.system()
        try:
            if system == "Windows":
                # Use wmic for Windows
                result = subprocess.run(
                    ["wmic", "path", "Win32_VideoController", "get", "AdapterRAM"],
                    capture_output=True,
                    text=True,
                    check=False, # Do not raise CalledProcessError
                    creationflags=subprocess.CREATE_NO_WINDOW # Hide console window
                )
                if result.returncode == 0 and "AdapterRAM" in result.stdout:
                    # Parse output, typically like: "AdapterRAM\n1073741824\n"
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip().isdigit():
                            ram_bytes = int(line.strip())
                            return int(ram_bytes / (1024**3))
            elif system == "Linux" or system == "Darwin": # Linux and macOS
                # Try nvidia-smi first
                if shutil.which("nvidia-smi"):
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    if result.returncode == 0:
                        vram_mib = int(result.stdout.strip().split('\n')[0])
                        return int(vram_mib / 1024)
                # Fallback for other GPUs or if nvidia-smi not found (e.g., AMD, Intel)
                # This is more complex and less reliable without specific tools.
                # For now, we'll log a warning and return None.
                logging.warning(
                    f"Could not detect GPU VRAM for {system}. "
                    "nvidia-smi not found or failed. "
                    "Manual detection for other GPUs (AMD/Intel) is not yet implemented. "
                    "Defaulting to configured max_vram_gb."
                )
            else:
                logging.warning(f"Unsupported operating system for GPU VRAM detection: {system}")
        except Exception as e:
            logging.warning(f"Error during GPU VRAM detection: {e}")
        return None

    def _extract_task_from_prompt(self, prompt: str) -> str:
        """
        Best-effort extraction of the task description from the orchestrator prompt.
        """
        match = re.search(r"TASK:\\s*(.+?)\\n\\nTESTS:", prompt, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return prompt.strip()

    def _get_model_info_from_benchmarks(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Loads model benchmark data from a JSON file. This simulates a "remote benchmark lookup".
        """
        if hasattr(self, "_benchmark_cache"):
            return self._benchmark_cache

        benchmark_file_path = self.project_root / ".autodev" / "lm_studio_model_benchmarks.json"
        if not benchmark_file_path.exists():
            logging.warning(f"Benchmark file not found: {benchmark_file_path}")
            self._benchmark_cache = {}
            return {}

        try:
            with open(benchmark_file_path, "r", encoding="utf-8") as f:
                self._benchmark_cache = json.load(f)
                return self._benchmark_cache
        except Exception as e:
            logging.error(f"Error loading model benchmarks from {benchmark_file_path}: {e}")
            self._benchmark_cache = {}
            return {}

    def _implement_dice_roller(self) -> str:
        """
        Create a /dice route and an accompanying Playwright test to verify
        the feature end-to-end. This provides a tangible feature when running
        AutoDev without external LLM connectivity.
        """
        app_dir = self.project_root / "crithappensdnd"
        if not app_dir.exists():
            return f"App directory not found at {app_dir}, no changes made."

        page_dir = app_dir / "src" / "routes" / "dice"
        page_dir.mkdir(parents=True, exist_ok=True)
        page_path = page_dir / "+page.svelte"
        test_path = app_dir / "tests" / "e2e" / "dice.spec.ts"
        test_path.parent.mkdir(parents=True, exist_ok=True)

        page_content = """<script lang=\\"ts\\">
  const diceOptions = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20', 'd100'];
  type Mode = 'normal' | 'advantage' | 'disadvantage';

  let selectedDie: string = 'd20';
  let diceCount = 1;
  let mode: Mode = 'normal';
  let resultSummary = 'No rolls yet.';
  let detail = 'Choose your dice, adjust the roll type, and click Roll.';
  let history: { summary: string; detail: string; at: number }[] = [];
  const maxHistory = 5;

  const clampCount = (value: number) => Math.min(10, Math.max(1, Math.floor(value || 1)));
  const rollDie = (sides: number) => Math.floor(Math.random() * sides) + 1;

  const formatMode = (value: Mode) => {
    if (value === 'advantage') return 'adv.';
    if (value === 'disadvantage') return 'disadv.';
    return 'normal';
  };

  function rollDice() {
    diceCount = clampCount(diceCount);
    const sides = Number(selectedDie.replace('d', '')) || 20;

    let summary = '';
    let description = '';

    if (mode !== 'normal' && sides === 20 && diceCount === 1) {
      const first = rollDie(20);
      const second = rollDie(20);
      const chosen = mode === 'advantage' ? Math.max(first, second) : Math.min(first, second);
      summary = `${chosen} on 1x${selectedDie} (${mode})`;
      description = `Rolls: ${first} vs ${second} -> ${chosen} (${mode})`;
    } else {
      const rolls: number[] = [];
      for (let i = 0; i < diceCount; i += 1) {
        rolls.push(rollDie(sides));
      }
      const total = rolls.reduce((sum, value) => sum + value, 0);
      summary = `${total} on ${diceCount}x${selectedDie}${mode !== 'normal' ? ` (${mode})` : ''}`;
      description = `${diceCount} × ${selectedDie}: ${rolls.join(' + ')} = ${total}`;
    }

    resultSummary = summary;
    detail = description;
    history = [{ summary, detail: description, at: Date.now() }, ...history].slice(0, maxHistory);
  }
</script>

<svelte:head>
  <title>Dice roller | Crit Happens</title>
  <meta name=\\"description\\" content=\\"Roll common D&D dice with advantage, disadvantage, and saved history.\\" />
</svelte:head>

<section class=\\"page-shell\\">
  <header class=\\"hero\\">
    <p class=\\"eyebrow\\">Tabletop utility</p>
    <h1>Dice roller</h1>
    <p class=\\"lede\\">
      Roll any common die, toggle advantage or disadvantage for d20 checks, and keep a short history of your most recent results.
    </p>
  </header>

  <div class=\\"card\\">
    <div class=\\"controls\\">
      <label>
        <span>Die</span>
        <select bind:value={selectedDie} data-testid=\\"die-select\\">
          {#each diceOptions as die}
            <option value={die}>{die}</option>
          {/each}
        </select>
      </label>

      <label>
        <span>Number of dice</span>
        <input
          type=\\"number\\"
          min=\\"1\\"
          max=\\"10\\"
          bind:value={diceCount}
          aria-label=\\"Number of dice\\"
          data-testid=\\"dice-count\\"
        />
      </label>

      <fieldset class=\\"mode\\">
        <legend>Roll type</legend>
        <label>
          <input type=\\"radio\\" name=\\"mode\\" value=\\"normal\\" bind:group={mode} />
          Normal
        </label>
        <label>
          <input type=\\"radio\\" name=\\"mode\\" value=\\"advantage\\" bind:group={mode} />
          Advantage
        </label>
        <label>
          <input type=\\"radio\\" name=\\"mode\\" value=\\"disadvantage\\" bind:group={mode} />
          Disadvantage
        </label>
      </fieldset>

      <button class=\\"roll\\" type=\\"button\\" on:click={rollDice}>
        Roll dice
      </button>
    </div>

    <div class=\\"result\\" data-testid=\\"roll-result\\">
      <p class=\\"label\\">Latest result</p>
      <p class=\\"value\\">{resultSummary}</p>
      <p class=\\"detail\\">{detail}</p>
    </div>
  </div>

  <div class=\\"history\\" data-testid=\\"roll-history\\">
    <div class=\\"history-header\\">
      <div>
        <p class=\\"eyebrow\\">Recent activity</p>
        <h2>Roll history</h2>
      </div>
      <span class=\\"pill\\">keeps last {maxHistory}</span>
    </div>
    {#if history.length === 0}
      <p class=\\"empty\\">No rolls yet. Take the dice for a spin!</p>
    {:else}
      <ul>
        {#each history as entry (entry.at)}
          <li>
            <div>
              <p class=\\"summary\\">{entry.summary}</p>
              <p class=\\"detail\\">{entry.detail}</p>
            </div>
            <span class=\\"timestamp\\">{new Date(entry.at).toLocaleTimeString()}</span>
          </li>
        {/each}
      </ul>
    {/if}
  </div>
</section>

<style>
  :global(body) {
    background: radial-gradient(circle at 20% 20%, #0f172a 0, #0b1021 50%, #070a16 100%);
    color: #e7edf6;
    font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
    min-height: 100vh;
  }

  .page-shell {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px 20px 64px;
    display: grid;
    gap: 24px;
  }

  .hero .eyebrow {
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 12px;
    color: #7dd3fc;
    margin: 0 0 4px 0;
  }

  .hero h1 {
    margin: 0;
    font-size: clamp(28px, 4vw, 36px);
  }

  .hero .lede {
    margin: 8px 0 0;
    color: #cbd5e1;
    max-width: 720px;
  }

  .card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(125, 211, 252, 0.06));
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 24px;
    display: grid;
    gap: 16px;
  }

  .controls {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    align-items: end;
  }

  label span,
  legend {
    display: block;
    font-size: 14px;
    color: #cbd5e1;
    margin-bottom: 6px;
  }

  select,
  input[type='number'] {
    width: 100%;
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.05);
    color: #e7edf6;
    font-size: 15px;
    outline: none;
  }

  select:focus,
  input[type='number']:focus {
    border-color: #7dd3fc;
    box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.25);
  }

  .mode {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.04);
  }

  .mode label {
    display: flex;
    gap: 8px;
    align-items: center;
    margin: 4px 0;
    color: #e7edf6;
  }

  .roll {
    background: linear-gradient(120deg, #38bdf8, #6366f1);
    color: #0b1021;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease;
  }

  .roll:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
  }

  .roll:active {
    transform: translateY(0);
  }

  .result {
    padding: 12px 14px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
  }

  .result .label {
    margin: 0;
    color: #93c5fd;
    font-size: 13px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .result .value {
    margin: 6px 0 4px;
    font-size: clamp(22px, 3vw, 28px);
    font-weight: 700;
  }

  .result .detail {
    margin: 0;
    color: #cbd5e1;
  }

  .history {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 18px;
    background: rgba(5, 8, 20, 0.9);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  }

  .history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .history h2 {
    margin: 0;
  }

  .pill {
    background: rgba(125, 211, 252, 0.15);
    color: #7dd3fc;
    border: 1px solid rgba(125, 211, 252, 0.4);
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 13px;
  }

  .history ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 10px;
  }

  .history li {
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
  }

  .history .summary {
    margin: 0 0 4px 0;
    font-weight: 600;
  }

  .history .detail,
  .history .timestamp {
    margin: 0;
    color: #94a3b8;
    font-size: 14px;
  }

  .empty {
    margin: 0;
    color: #cbd5e1;
  }

  @media (max-width: 640px) {
    .controls {
      grid-template-columns: 1fr;
    }

    .history li {
      flex-direction: column;
      align-items: flex-start;
    }

    .history .timestamp {
      align-self: flex-end;
    }
  }
</style>
"""

        test_content = """import { expect, test } from '@playwright/test';

test('dice roller supports basic rolls and history', async ({ page }) => {
  await page.goto('/dice');

  await expect(page.getByRole('heading', { name: /dice roller/i })).toBeVisible();
  await page.getByLabel('Number of dice').fill('2');
  await page.getByRole('button', { name: 'Roll dice' }).click();

  const result = page.getByTestId('roll-result');
  await expect(result).toContainText('Result');
  const history = page.getByTestId('roll-history');
  await expect(history.getByRole('listitem')).toHaveCount(1);

  await page.getByLabel('Advantage').check();
  await page.getByRole('button', { name: 'Roll dice' }).click();
  await expect(history.getByRole('listitem')).toHaveCount(2);

  await page.getByLabel('Disadvantage').check();
  await page.getByRole('button', { name: 'Roll dice' }).click();
  await expect(history.getByRole('listitem')).toHaveCount(3);
});
"""

        page_path.write_text(page_content.replace('\\"', '"'), encoding="utf-8")
        test_path.write_text(test_content.replace('\\"', '"'), encoding="utf-8")

        return (
            "Offline mode: created dice roller feature with UI and e2e test.\n"
            f"- Page: {page_path}\n"
            f"- Test: {test_path}"
        )
