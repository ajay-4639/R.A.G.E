# RAG OS Docker Image
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY rag_os/ ./rag_os/

# Install the package with API dependencies
RUN pip install --no-cache-dir ".[api]"

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security
RUN groupadd -r ragos && useradd -r -g ragos ragos

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY rag_os/ ./rag_os/
COPY pyproject.toml ./

# Create data directories
RUN mkdir -p /app/data /app/indexes /app/logs && \
    chown -R ragos:ragos /app

# Switch to non-root user
USER ragos

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_OS_DATA_DIR=/app/data \
    RAG_OS_INDEX_DIR=/app/indexes \
    RAG_OS_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command: run API server
CMD ["python", "-m", "uvicorn", "rag_os.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
