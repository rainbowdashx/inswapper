FROM alpine/git:2.36.2 as download

RUN git lfs install
RUN git clone https://huggingface.co/spaces/sczhou/CodeFormer --depth 1 /CodeFormer

RUN rm -rf /CodeFormer/.git

RUN apk add --no-cache wget
RUN wget -q -O /inswapper_128.onnx https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true



FROM python:3.10.9-slim

RUN python -m venv venv

RUN mkdir -p /app/handler
RUN mkdir -p /app/checkpoints

RUN apt-get update && apt-get install -y build-essential

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y wget

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt requirements.txt
COPY *.py ./
COPY handler/ ./handler
COPY src .

RUN pip install git+https://github.com/sajjjadayobi/FaceLib.git
RUN pip install basicsr



RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functiona import rgb_to_grayscale/' \
    /usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py

COPY --from=download /CodeFormer /app/CodeFormer
COPY --from=download /inswapper_128.onnx /app/checkpoints/inswapper_128.onnx

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r CodeFormer/requirements.txt

RUN python handler/cache.py

RUN apt-get update && apt-get install -y dos2unix && dos2unix /app/start.sh

RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]

