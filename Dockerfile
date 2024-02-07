FROM python:3.10.9-slim

RUN mkdir -p /app/handler

WORKDIR /app

COPY requirements.txt requirements.txt
COPY *.py ./
COPY handler/ ./handler



RUN apk add --no-cache wget && \
    wget -q -O ./handler/inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true

RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir -r requirements.txt

RUN python handler/cache.py

