import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc


class InferenceServiceImpl(inference_pb2_grpc.InferenceServiceServicer):
    """
    This class implements the methods defined in our .proto file.
    """

    def TranscribeStream(self, request_iterator, context):
        for audio_chunk in request_iterator:
            print(f"-> Received audio chunk of size {len(audio_chunk.data)} bytes")

        print(f"-> Transcribing audio stream")
        return inference_pb2.TextResponse(text="Transcribed text")

    def FormatText(self, request, context):
        print(f"-> Formatting text: {request.text}")
        return inference_pb2.TextResponse(text="Formatted text")

    def TranscribeAndFix(self, request_iterator, context):
        for audio_chunk in request_iterator:
            print(f"-> Received audio chunk of size {len(audio_chunk.data)} bytes")

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented")
        return inference_pb2.TextResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServiceImpl(), server
    )
    port = "50051"
    server.add_insecure_port(f"[::]:{port}")
    print(f"Server starting on port {port}")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")


if __name__ == "__main__":
    serve()
