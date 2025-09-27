import logging
from typing import Optional
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np
import ruptures as rpt
from config import config
import librosa

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
            )
            self.pipeline = BatchedInferencePipeline(
                model=self.model,
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def extract_enhanced_features(self, audio_array, sample_rate=16000):
        """
        Extract comprehensive features for language change detection
        """
        hop_length = 160  # 10ms frames

        # 1. Enhanced MFCC (more coefficients + deltas)
        mfcc = librosa.feature.mfcc(
            y=audio_array,
            sr=sample_rate,
            n_mfcc=20,  # Increased from 13
            hop_length=hop_length,
            n_fft=512,
        )

        # Delta and delta-delta MFCC
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # 2. Spectral features for language discrimination
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_array, sr=sample_rate, hop_length=hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_array, sr=sample_rate, hop_length=hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_array, sr=sample_rate, hop_length=hop_length
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio_array, hop_length=hop_length
        )

        # 3. Chroma features (pitch class distribution)
        chroma = librosa.feature.chroma_stft(
            y=audio_array, sr=sample_rate, hop_length=hop_length
        )

        # Combine all features
        features = np.vstack(
            [
                mfcc,
                mfcc_delta,
                mfcc_delta2,
                spectral_centroid,
                spectral_rolloff,
                spectral_bandwidth,
                zero_crossing_rate,
                chroma,
            ]
        )

        return features.T  # Shape: (n_frames, n_features)

    def detect_language_changes_using_pelt(
        self,
        audio_array: np.ndarray,
        sample_rate=config.AUDIO_TARGET_SAMPLE_RATE,
        penalty=5,
        min_size=1,
    ):
        hop_length = 160  # 10ms frames at 16kHz (160 samples)
        features = self.extract_enhanced_features(audio_array, sample_rate)
        min_size_frames = int(min_size * sample_rate / hop_length)
        algo = rpt.Pelt(model="l2", min_size=min_size_frames).fit(features)
        change_points_frames = algo.predict(pen=penalty)
        change_points_seconds = []
        for cp in change_points_frames[:-1]:
            timestamp = cp * hop_length / sample_rate
            change_points_seconds.append(timestamp)

        return change_points_seconds

    def create_audio_segments(
        self,
        audio_array: np.ndarray,
        change_points: list[float],
        sample_rate=config.AUDIO_TARGET_SAMPLE_RATE,
    ) -> list[dict]:
        segments = []
        if not change_points or change_points[0] > 0:
            change_points = [0.0] + change_points

        audio_duration = len(audio_array) / sample_rate
        if not change_points or change_points[-1] < audio_duration:
            change_points.append(audio_duration)

        for i in range(len(change_points) - 1):
            start_time = change_points[i]
            end_time = change_points[i + 1]
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_array[start_sample:end_sample]
            segments.append(
                {
                    "audio": segment_audio,
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                }
            )

        return segments

    def transcribe_segments_with_faster_whisper(self, segments) -> str:
        transcription = ""
        for segment in segments:
            segments_result, _ = self.model.transcribe(
                segment["audio"],
                language=None,
                vad_filter=True,
            )
            transcription_text = ""
            for seg in segments_result:
                transcription_text += seg.text + " "

            transcription += transcription_text + " "

        return transcription.strip()

    def multilingual_transcribe_with_pelt(
        self, audio_array: np.ndarray, sample_rate=config.AUDIO_TARGET_SAMPLE_RATE
    ) -> str:
        print(f"Audio duration: {len(audio_array) / sample_rate:.2f} seconds")
        print("Detecting language change points...")
        change_points = self.detect_language_changes_using_pelt(
            audio_array,
        )
        print(f"Detected {len(change_points)} change points at: {change_points}")
        print("Creating audio segments...")
        segments = self.create_audio_segments(audio_array, change_points)
        print(f"Created {len(segments)} segments")
        print("Transcribing segments...")
        transcription = self.transcribe_segments_with_faster_whisper(segments)
        print(f"Transcription: {transcription}")
        return transcription

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
