import grpc
from concurrent import futures
import time
import logging
import argparse
import sys
import os
import inference_pb2_grpc
from services.inference_service import InferenceServiceImpl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def serve(port: int = 50051, model_size: str = "base", device: str = "cpu"):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_service = InferenceServiceImpl(model_size=model_size, device=device)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(inference_service, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(
        f"Starting server on {listen_addr} with model '{model_size}' on '{device}'"
    )
    server.start()
    logger.info("Server started successfully")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace_period=5)
        logger.info("Server stopped")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Audio Inference gRPC Server")
    parser.add_argument(
        "--port", type=int, default=50051, help="Port to listen on (default: 50051)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run on (default: cpu)",
    )
    args = parser.parse_args()
    logger.info(f"Starting inference server with configuration:")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    serve(port=args.port, model_size=args.model, device=args.device)


if __name__ == "__main__":
    main()
