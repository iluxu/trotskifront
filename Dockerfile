# Use official Python base image
FROM python:3.10-slim

WORKDIR /app

# System deps for audio/whisper models
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3-av \
    pkg-config \
    gcc \
    g++ \
    make \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir "cython<3" && \
    pip install --no-cache-dir --no-deps faster-whisper==0.10.1 && \
    pip install --no-cache-dir \
      numpy==1.26.4 \
      websockets==12.0 \
      openai==2.0.1 \
      httpx==0.27.2 \
      python-dotenv==1.0.1 \
      requests==2.32.3 \
      "ctranslate2<4,>=3.22" \
      "onnxruntime<2,>=1.14" \
      "tokenizers<0.16,>=0.13" \
      "huggingface-hub>=0.13"

COPY optimized_stt_server_v3.py ./

ENV STT_HOST=0.0.0.0
ENV STT_PORT=8123
ENV STT_DEVICE=cpu
ENV STT_COMPUTE=int8

EXPOSE 8123

CMD ["python", "optimized_stt_server_v3.py"]
