import logging
from typing import Optional
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np
from config import config

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.pipeline: Optional[BatchedInferencePipeline] = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(
                f"Loading Whisper model: {config.WHISPER_MODEL_SIZE} on {config.WHISPER_DEVICE}"
            )
            self.model = WhisperModel(
                config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
                download_root=config.WHISPER_DOWNLOAD_ROOT,
            )
            self.pipeline = BatchedInferencePipeline(
                model=self.model,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe_audio(self, pcm_audio: np.ndarray) -> str:
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")

        try:
            segments, info = self.pipeline.transcribe(
                pcm_audio,
                beam_size=5,
                language=None,
                vad_filter=True,
                task="transcribe",
                multilingual=True,
                language_detection_threshold=0.7,
            )
            transcription_text = ""
            for segment in segments:
                transcription_text += segment.text + " "

            transcription_text = transcription_text.strip()
            logger.info(
                f"Transcription completed: '{transcription_text[:100]}...' (detected language: {info.language})"
            )
            return transcription_text

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
