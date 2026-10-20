FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies needed for compiling and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories for audio storage and artifacts
RUN mkdir -p /app/audio /app/artifacts

# Copy application source, migrations, artifacts, and configuration
COPY alembic.ini .
COPY migrations/ ./migrations/
COPY artifacts/ ./artifacts/
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run migrations and start uvicorn
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.api:app --host 0.0.0.0 --port 8000"]
