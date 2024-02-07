FROM alpine/git:2.36.2 as download

RUN apk add --no-cache wget
RUN wget -q -O /inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true



FROM python:3.10.9-slim

RUN mkdir -p /app/handler
RUN mkdir -p /app/checkpoints

RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y wget

RUN apt-get update && apt-get install -y dos2unix && dos2unix /start.sh

WORKDIR /app

COPY requirements.txt requirements.txt
COPY *.py ./
COPY handler/ ./handler

COPY --from=download /inswapper_128.onnx /app/checkpoints/inswapper_128.onnx

RUN pip install --no-cache-dir -r requirements.txt

RUN python handler/cache.py

COPY src .
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]

