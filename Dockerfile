# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1  # Prevent Python from writing .pyc files
ENV PYTHONUNBUFFERED=1         # Ensure Python output is not buffered
ENV APP_HOME=/app

# Set working directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libffi-dev \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports (adjust based on your application, e.g., FastAPI default is 8000)
EXPOSE 8000

# Define default entrypoint
ENTRYPOINT ["uvicorn"]

# Start the application (replace app.main:app with your FastAPI app's module and object)
CMD ["src.inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]