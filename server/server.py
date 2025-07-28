import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
import io
import tempfile
from faster_whisper import WhisperModel, BatchedInferencePipeline
import traceback

MODEL_SIZE = "small"


class InferenceServiceImpl(inference_pb2_grpc.InferenceServiceServicer):
    """
    This class implements the methods defined in our .proto file.
    """

    def __init__(self, whisper_model: WhisperModel):
        self.whisper_model = whisper_model

    def _transcribe_audio(self, audio_chunks: bytes) -> str:
        try:
            audio_bytes_io = io.BytesIO()
            for audio_chunk in audio_chunks:
                print(type(audio_chunk.audio_bytes))
                audio_bytes_io.write(audio_chunk.audio_bytes)

            with tempfile.NamedTemporaryFile(delete=True, suffix="webm") as temp_file:
                temp_file.write(audio_bytes_io.getvalue())
                temp_file.flush()

                batched_model = BatchedInferencePipeline(model=self.whisper_model)
                segments, info = batched_model.transcribe(
                    temp_file.name,
                    batch_size=16,
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

    def TranscribeStream(self, request_iterator: bytes, context: grpc.ServicerContext):
        print(f"-> Transcribing audio stream")
        transcribed_text = self._transcribe_audio(request_iterator)
        return inference_pb2.TextResponse(text=transcribed_text)

    def FormatText(
        self, request: inference_pb2.TextRequest, context: grpc.ServicerContext
    ):
        print(f"-> Formatting text: {request.text}")
        return inference_pb2.TextResponse(text="Formatted text")

    def TranscribeAndFix(self, request_iterator, context):
        transcribed_text = self._transcribe_audio(request_iterator)
        return inference_pb2.TextResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print(f"Whisper model loaded")

    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServiceImpl(whisper_model), server
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
