FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install grpcio-tools \
    && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

COPY . .
RUN mkdir -p server/gen
RUN python -m grpc_tools.protoc \
    -I. \
    --python_out=server/gen \
    --grpc_python_out=server/gen \
    protos/inference.proto

FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY --from=builder /app /app
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

EXPOSE 50051
CMD ["python", "main.py"]