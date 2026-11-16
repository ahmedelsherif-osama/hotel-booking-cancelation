
FROM python:3.11-slim

WORKDIR /app

# CatBoost runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy API and trained model
COPY predict.py .
COPY models/ ./models/

# Render provides PORT; use 9696 locally by default
ENV PORT=9696

EXPOSE 9696

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
