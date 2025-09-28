# Production Dockerfile for RxVision2025
# Multi-stage build for optimized production image

# Build stage
FROM python:3.10-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels for metadata
LABEL maintainer="RxVision2025 Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="rxvision2025" \
      org.label-schema.description="Advanced pharmaceutical computer vision system" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY pyproject.toml README.md ./

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.10-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r rxvision && useradd -r -g rxvision -u 1000 rxvision

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=rxvision:rxvision src/ ./src/
COPY --chown=rxvision:rxvision scripts/ ./scripts/
COPY --chown=rxvision:rxvision configs/ ./configs/
COPY --chown=rxvision:rxvision pyproject.toml README.md ./

# Create necessary directories
RUN mkdir -p logs models data reports && \
    chown -R rxvision:rxvision /app

# Install the package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER rxvision

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Default command
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]