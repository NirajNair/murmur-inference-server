import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self):
        self._load_config()

    def _load_config(self):
        # Server Configuration
        self.SERVER_PORT = self._get_required_int("SERVER_PORT")
        self.SERVER_HOST = self._get_required_str("SERVER_HOST")
        self.MAX_WORKERS = self._get_required_int("MAX_WORKERS")
        self.GRACE_PERIOD = self._get_required_int("GRACE_PERIOD")

        # Model Configuration
        self.WHISPER_MODEL_SIZE = self._get_required_str("WHISPER_MODEL_SIZE")
        self.WHISPER_DEVICE = self._get_required_str("WHISPER_DEVICE")
        self.WHISPER_COMPUTE_TYPE = self._get_required_str("WHISPER_COMPUTE_TYPE")
        self.WHISPER_DOWNLOAD_ROOT = self._get_required_str("WHISPER_DOWNLOAD_ROOT")
        self.LLM_MODEL_PATH = self._get_required_str("LLM_MODEL_PATH")
        self.LLM_N_CTX = self._get_required_int("LLM_N_CTX")
        self.LLM_N_THREADS = self._get_required_int("LLM_N_THREADS")

        # Audio Processing Configuration
        self.AUDIO_TARGET_SAMPLE_RATE = self._get_required_int(
            "AUDIO_TARGET_SAMPLE_RATE"
        )
        self.AUDIO_TARGET_CHANNELS = self._get_required_int("AUDIO_TARGET_CHANNELS")
        self.AUDIO_DEBUG_DIR = self._get_required_str("AUDIO_DEBUG_DIR")
        self.AUDIO_ZERO_THRESHOLD = self._get_required_float("AUDIO_ZERO_THRESHOLD")
        self.AUDIO_CHUNK_SIZE_THRESHOLD = self._get_required_int(
            "AUDIO_CHUNK_SIZE_THRESHOLD"
        )
        self.AUDIO_HEADER_SIZE = self._get_required_int("AUDIO_HEADER_SIZE")

        # Text Correction Configuration
        self.LLM_POOL_SIZE = self._get_required_int("LLM_POOL_SIZE")
        self.LLM_MAX_TOKENS = self._get_required_int("LLM_MAX_TOKENS")
        self.LLM_TEMPERATURE = self._get_required_float("LLM_TEMPERATURE")
        self.LLM_TOP_P = self._get_required_float("LLM_TOP_P")
        self.LLM_TOP_K = self._get_required_int("LLM_TOP_K")
        self.LLM_N_BATCH = self._get_required_int("LLM_N_BATCH")
        self.LLM_N_UBATCH = self._get_required_int("LLM_N_UBATCH")
        self.LLM_CORRECTION_THRESHOLD = self._get_required_float(
            "LLM_CORRECTION_THRESHOLD"
        )
        self.LLM_REQUEST_TIMEOUT = self._get_required_float("LLM_REQUEST_TIMEOUT")

        # Logging Configuration
        self.LOG_LEVEL = self._get_required_str("LOG_LEVEL")
        self.LOG_FORMAT = self._get_required_str("LOG_FORMAT")

        # Performance Configuration
        self.ENABLE_DEBUG_AUDIO = self._get_required_bool("ENABLE_DEBUG_AUDIO")

    def _get_required_str(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value

    def _get_required_int(self, key: str) -> int:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"Environment variable '{key}' must be an integer, got: {value}"
            )

    def _get_required_float(self, key: str) -> float:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"Environment variable '{key}' must be a float, got: {value}"
            )

    def _get_required_bool(self, key: str) -> bool:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return value.lower() == "true"

    def setup_logging(self):
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format=self.LOG_FORMAT)

    def get_llm_model_path(self) -> str:
        if os.path.isabs(self.LLM_MODEL_PATH):
            return self.LLM_MODEL_PATH
        return os.path.join(os.path.dirname(__file__), self.LLM_MODEL_PATH)

    def get_audio_debug_dir(self) -> str:
        if os.path.isabs(self.AUDIO_DEBUG_DIR):
            return self.AUDIO_DEBUG_DIR
        return os.path.join(os.path.dirname(__file__), "..", self.AUDIO_DEBUG_DIR)


config = Config()
