import grpc
from concurrent import futures
import time
import logging
import argparse

from server import inference_pb2_grpc
from services.inference_service import InferenceServiceImpl
from config import config

config.setup_logging()
logger = logging.getLogger(__name__)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS))
    inference_service = InferenceServiceImpl()
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(inference_service, server)
    listen_addr = f"[{config.SERVER_HOST}]:{config.SERVER_PORT}"
    server.add_insecure_port(listen_addr)
    logger.info(
        f"Starting server on {listen_addr} with model '{config.WHISPER_MODEL_SIZE}' on '{config.WHISPER_DEVICE}'"
    )
    server.start()
    logger.info("Server started successfully")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace_period=config.GRACE_PERIOD)
        logger.info("Server stopped")


def main():
    parser = argparse.ArgumentParser(description="Audio Inference gRPC Server")
    parser.add_argument(
        "--port",
        type=int,
        default=config.SERVER_PORT,
        help=f"Port to listen on (default: {config.SERVER_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.WHISPER_MODEL_SIZE,
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help=f"Whisper model size (default: {config.WHISPER_MODEL_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.WHISPER_DEVICE,
        choices=["cpu", "cuda", "auto"],
        help=f"Device to run on (default: {config.WHISPER_DEVICE})",
    )
    args = parser.parse_args()
    logger.info(f"Starting inference server with configuration:")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {args.device}")
    serve()


if __name__ == "__main__":
    main()
