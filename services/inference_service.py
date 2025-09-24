import logging
from typing import Iterator, Optional
from faster_whisper import WhisperModel
from llama_cpp import Llama
import grpc
from concurrent import futures
import os
from server import inference_pb2
from server import inference_pb2_grpc
from services.audio_utils import (
    universal_audio_to_pcm,
    inspect_audio_bytes,
    reconstruct_wav_from_chunks,
)
from config import config

logger = logging.getLogger(__name__)


class InferenceServiceImpl(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.model: Optional[WhisperModel] = None
        self.llm_model: Optional[Llama] = None
        self._load_models()

    def _load_models(self):
        try:
            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self.model = WhisperModel(self.model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

        try:
            llm_model_path = config.get_llm_model_path()
            logger.info(f"Loading LLM model from: {llm_model_path}")
            self.llm_model = Llama(
                model_path=llm_model_path,
                n_ctx=config.LLM_N_CTX,
                n_threads=config.LLM_N_THREADS,
                verbose=False,
            )
            logger.info("LLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            logger.warning("Text correction will fall back to basic formatting")
            self.llm_model = None

    def _create_correction_prompt(self, text: str) -> str:
        """Create a Wispr Flow-inspired prompt for text correction."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            You are a expert text correction assistant specialized in fixing speech-to-text transcription errors. Your task is to correct grammar, spelling, punctuation, and formatting while preserving the original meaning and intent.

            **Primary Corrections:**
            • Fix spelling mistakes and speech recognition errors (their/there, to/too, etc.)
            • Add proper punctuation (periods, commas, question marks, exclamation points)
            • Correct capitalization (sentence beginnings, proper nouns)
            • Break up run-on sentences into readable segments
            • Remove excessive filler words only when they impair readability (um, uh, like, you know)
            • Fix grammar while maintaining natural speech patterns
            • Correct word boundaries and missing spaces

            **Smart Formatting Commands:**
            When you detect dictation commands with 80%+ confidence based on context, convert them appropriately:
            • "period" / "full stop" → .
            • "comma" → ,
            • "question mark" → ?
            • "exclamation point" / "exclamation mark" → !
            • "new line" / "new paragraph" → line break
            • "bullet point" / "dash" → • (when creating lists)
            • "number one, number two" → 1. 2. (when creating numbered lists)
            • "bold [text]" / "italic [text]" → apply only if clearly intentional formatting

            **Quality Guidelines:**
            • Preserve the speaker's voice, style, and personality
            • Keep technical terms, jargon, and proper nouns intact
            • Don't over-correct casual or conversational language
            • Maintain regional dialects and speech patterns where appropriate
            • If multiple interpretations are possible, choose the most contextually appropriate
            • Handle multilingual content and code-switching naturally
            • If the text looks like an email then format it as an email

            **Critical Rules:**
            • Return ONLY the corrected text - no explanations, comments, markdown, suffix, prefix or any other text.
            • Output plain text format only
            • Preserve the core message and meaning exactly
            • When uncertain about a correction, err on the side of minimal changes

            <|eot_id|><|start_header_id|>user<|end_header_id|>

            Please correct this transcribed text: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            """

    def _correct_text_with_llm(self, text: str) -> str:
        if not self.llm_model or not text.strip():
            return text

        try:
            prompt = self._create_correction_prompt(text)
            result = self.llm_model(
                prompt,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                top_p=config.LLM_TOP_P,
                stop=["<|eot_id|>", "\n\n"],
                echo=False,
            )
            corrected_text = result["choices"][0]["text"].strip()
            if (
                not corrected_text
                or len(corrected_text) < len(text) * config.LLM_CORRECTION_THRESHOLD
            ):
                logger.warning(
                    "LLM correction resulted in unusually short text, using original"
                )
                return text
            return corrected_text

        except Exception as e:
            logger.error(f"LLM text correction failed: {str(e)}")
            return text

    def TranscribeStream(
        self, request_iterator: Iterator[inference_pb2.AudioChunk], context
    ) -> inference_pb2.TextResponse:
        try:
            audio_chunks = []
            total_bytes = 0
            for chunk in request_iterator:
                if chunk.audio_bytes:
                    audio_chunks.append(chunk.audio_bytes)
                    total_bytes += len(chunk.audio_bytes)

            logger.info(
                f"Received {len(audio_chunks)} audio chunks, total bytes: {total_bytes}"
            )

            if not audio_chunks:
                return inference_pb2.TextResponse(text="")

            try:
                combined_audio_bytes = reconstruct_wav_from_chunks(audio_chunks)
                inspection = inspect_audio_bytes(combined_audio_bytes)
                logger.info(f"Reconstructed audio inspection: {inspection}")
            except ValueError as e:
                logger.error(f"Failed to reconstruct WAV file: {e}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Failed to reconstruct WAV file from chunks.")
                return inference_pb2.TextResponse(text="")

            try:
                pcm_audio = universal_audio_to_pcm(
                    combined_audio_bytes,
                    target_sample_rate=config.AUDIO_TARGET_SAMPLE_RATE,
                )
                logger.info(
                    f"Converted to PCM: {pcm_audio.shape} samples, duration: {len(pcm_audio)/config.AUDIO_TARGET_SAMPLE_RATE:.2f}s"
                )
            except ValueError as e:
                logger.error(f"Universal audio conversion failed: {str(e)}")
                logger.error(f"Audio data details:")
                logger.error(f"  - Length: {len(combined_audio_bytes)} bytes")
                logger.error(f"  - Number of chunks: {len(audio_chunks)}")

                if len(audio_chunks) > 0:
                    chunk_sizes = [len(chunk) for chunk in audio_chunks]
                    logger.error(f"  - Chunk sizes: {chunk_sizes}")
                    logger.error(
                        f"  - Average chunk size: {sum(chunk_sizes)/len(chunk_sizes):.0f} bytes"
                    )

                logger.error(
                    f"  - First 32 bytes (hex): {combined_audio_bytes[:32].hex()}"
                )
                logger.error(f"  - Inspection result: {inspection}")
                if inspection.get("zero_percentage", 0) > config.AUDIO_ZERO_THRESHOLD:
                    logger.error("🚨 CLIENT ISSUE DETECTED:")
                    logger.error("   Your client is sending mostly/all zero bytes!")
                    logger.error("   This suggests:")
                    logger.error("     - Audio recording is not working")
                    logger.error("     - WAV file reading failed")
                    logger.error("     - Data corruption during transmission")
                    logger.error("   See server logs for debugging suggestions.")

                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"Audio conversion failed: {str(e)}")
                return inference_pb2.TextResponse(text="")

            try:
                segments, info = self.model.transcribe(pcm_audio, language=None)
                transcription_text = ""
                for segment in segments:
                    transcription_text += segment.text + " "

                transcription_text = transcription_text.strip()
                logger.info(
                    f"Transcription completed: '{transcription_text[:100]}...' (detected language: {info.language})"
                )

                return inference_pb2.TextResponse(text=transcription_text)

            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Transcription failed: {str(e)}")
                return inference_pb2.TextResponse(text="")

        except Exception as e:
            logger.error(f"TranscribeStream error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return inference_pb2.TextResponse(text="")

    def FormatText(
        self, request: inference_pb2.TextRequest, context
    ) -> inference_pb2.TextResponse:
        try:
            input_text = request.text
            if not input_text or not input_text.strip():
                return inference_pb2.TextResponse(text="")

            logger.info(
                f"FormatText received: '{input_text[:100]}{'...' if len(input_text) > 100 else ''}'"
            )
            if self.llm_model:
                try:
                    corrected_text = self._correct_text_with_llm(input_text)
                    logger.info(
                        f"LLM correction completed: '{corrected_text[:100]}{'...' if len(corrected_text) > 100 else ''}'"
                    )
                    return inference_pb2.TextResponse(text=corrected_text)
                except Exception as e:
                    logger.error(
                        f"LLM correction failed, falling back to basic formatting: {str(e)}"
                    )

            formatted_text = input_text.strip()
            if formatted_text and not formatted_text.endswith((".", "!", "?")):
                formatted_text += "."

            if formatted_text:
                formatted_text = formatted_text[0].upper() + formatted_text[1:]

            logger.info(f"Basic formatting completed: '{formatted_text}'")
            return inference_pb2.TextResponse(text=formatted_text)

        except Exception as e:
            logger.error(f"FormatText error: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Text formatting failed: {str(e)}")
            return inference_pb2.TextResponse(text="")

    def TranscribeAndFix(
        self, request_iterator: Iterator[inference_pb2.AudioChunk], context
    ) -> inference_pb2.TextResponse:
        transcription_response = self.TranscribeStream(request_iterator, context)
        text_request = inference_pb2.TextRequest(text=transcription_response.text)
        return self.FormatText(text_request, context)
