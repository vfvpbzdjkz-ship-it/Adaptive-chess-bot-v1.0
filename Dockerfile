FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch is much smaller (~250MB vs 2GB+)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir chess numpy requests rich

COPY . .

RUN mkdir -p data/models data/buffer data/logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# data/ is mounted as a persistent volume in Railway
VOLUME ["/app/data"]

CMD ["python", "main.py"]
