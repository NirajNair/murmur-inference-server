import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
import io
import tempfile
from faster_whisper import WhisperModel, BatchedInferencePipeline
import traceback
from llama_cpp import Llama

WHISPER_MODEL_SIZE = "small"
LLM_MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"


class InferenceServiceImpl(inference_pb2_grpc.InferenceServiceServicer):
    """
    This class implements the methods defined in our .proto file.
    """

    def __init__(self, whisper_model: WhisperModel, llm_model: Llama):
        self.whisper_model = whisper_model
        self.batched_model = BatchedInferencePipeline(model=self.whisper_model)
        self.llm_model = llm_model

    def _transcribe_audio(self, audio_chunks: bytes) -> str:
        try:
            audio_bytes_io = io.BytesIO()
            for audio_chunk in audio_chunks:
                print(type(audio_chunk.audio_bytes))
                audio_bytes_io.write(audio_chunk.audio_bytes)

            with tempfile.NamedTemporaryFile(delete=True, suffix="webm") as temp_file:
                temp_file.write(audio_bytes_io.getvalue())
                temp_file.flush()

                segments, info = self.batched_model.transcribe(
                    temp_file.name,
                    batch_size=32,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                transcribed_text = ""
                for segment in segments:
                    transcribed_text += segment.text

                return transcribed_text
        except Exception as e:
            print(f"-> Error transcribing audio: {repr(e)}")
            traceback.print_exc()
            return ""

    def _format_text(self, text: str) -> str:
        try:
            if not text:
                return ""

            messages = [
                {
                    "role": "system",
                    "content": """
                        You are an expert AI text formatter. Your task is to transform transcribed speech into polished, professional written text.

                        Core Instructions:
                            1. Punctuation: Add correct punctuation (periods, commas, question marks, exclamation marks, colons, semicolons)
                            2. Structure: Format according to specified context (email, message, documentation, etc.) only when explicitly indicated
                            3. Commands: Execute clear formatting commands like "new line," "new paragraph," "bullet point," "numbered list," "heading," then remove the command phrases from output
                            4. Content Preservation: Distinguish between formatting commands and actual content—when uncertain, treat as regular text
                            5. Error Correction: Fix obvious transcription errors while preserving original intent and tone
                            6. Filler Removal: Remove speech disfluencies like "um," "uh," "hmm," "you know," etc.
                            7. Grammar: Correct grammatical errors only when certain they're mistakes

                        Critical Rules (Non-negotiable):
                            1. Output ONLY the formatted text—no explanations, prefixes, or suffixes
                            2. Use only the provided transcribed input—do not add invented content
                            3. Treat each input as a complete unit requiring formatting
                            4. When context is ambiguous, default to standard sentence formatting
                            5. Preserve the speaker's voice and intended meaning

                        Example:
                        Input: "um hey john can you send me that report by friday thanks"
                        Output: "Hey John, can you send me that report by Friday? Thanks."
                    """,
                },
                {
                    "role": "user",
                    "content": text,
                },
            ]
            response = self.llm_model.create_chat_completion(
                messages=messages, max_tokens=1024, temperature=0.3
            )
            formatted_text = response["choices"][0]["message"]["content"]
            return formatted_text
        except Exception as e:
            print(f"-> Error formatting text: {repr(e)}")
            traceback.print_exc()
            return ""

    def TranscribeStream(self, request_iterator: bytes, context: grpc.ServicerContext):
        print(f"-> Transcribing audio stream")
        transcribed_text = self._transcribe_audio(request_iterator)
        return inference_pb2.TextResponse(text=transcribed_text)

    def FormatText(
        self, request: inference_pb2.TextRequest, context: grpc.ServicerContext
    ):
        formatted_text = self._format_text(request.text)
        return inference_pb2.TextResponse(text=formatted_text)

    def TranscribeAndFix(self, request_iterator, context):
        transcribed_text = self._transcribe_audio(request_iterator)
        formatted_text = self._format_text(transcribed_text)
        return inference_pb2.TextResponse(text=formatted_text)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        num_workers=5,
    )
    print(f"Whisper model loaded")

    llm_model = Llama(
        model_path=LLM_MODEL_PATH,
        chat_format="llama-3",
        n_ctx=2048,
        n_threads=8,
        n_batch=256,
        n_gpu_layers=35,
        metal=True,
        verbose=False,
        f16_kv=True,
        use_mlock=True,
        use_mmap=True,
    )
    print(f"LLM model loaded")

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServiceImpl(whisper_model, llm_model), server
    )

    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Server started on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")


if __name__ == "__main__":
    serve()
