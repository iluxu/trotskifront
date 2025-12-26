FROM python:3.10-slim

WORKDIR /app

COPY requirements.fly.txt ./
RUN pip install --no-cache-dir -r requirements.fly.txt

COPY optimized_stt_server_v3.py ./

ENV STT_HOST=0.0.0.0
ENV STT_PORT=8123
ENV STT_BACKEND=openai
ENV STT_STT_MODEL=whisper-1

EXPOSE 8123

CMD ["python", "optimized_stt_server_v3.py"]
