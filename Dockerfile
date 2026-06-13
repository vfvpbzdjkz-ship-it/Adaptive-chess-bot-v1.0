FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first (avoids pulling the 2GB CUDA build from PyPI)
RUN pip install --no-cache-dir "torch==2.2.2" \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else (torch already satisfied above, pip skips it)
RUN pip install --no-cache-dir \
    "chess==1.10.0" \
    "numpy==1.26.4" \
    "requests==2.31.0" \
    "rich==13.7.1" \
    "huggingface_hub>=0.22.0" \
    "google-auth>=2.28.0" \
    "google-api-python-client>=2.120.0"

COPY . .

RUN mkdir -p data/models data/buffer data/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
