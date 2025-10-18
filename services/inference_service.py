import logging
from typing import Iterator
import grpc
from server import inference_pb2
from server import inference_pb2_grpc
from services.audio_utils import (
    universal_audio_to_pcm,
    inspect_audio_bytes,
    reconstruct_wav_from_chunks,
)
from services.transcription_service import TranscriptionService
from services.llm_service import LLMService
from config import config

logger = logging.getLogger(__name__)


class InferenceServiceImpl(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        self.transcription_service = TranscriptionService()
        self.llm_service = LLMService()

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
                transcription_text = (
                    self.transcription_service.transcribe_audio_with_groq(
                        audio_chunks
                    )
                )
                return inference_pb2.TextResponse(text=transcription_text)
            except Exception as e:
                logger.error(str(e))

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
                transcription_text = self.transcription_service.transcribe_audio(
                    pcm_audio
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
            if self.llm_service.is_available():
                try:
                    corrected_text = self.llm_service.correct_text(input_text)
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
