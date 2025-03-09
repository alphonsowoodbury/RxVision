# Development Setup

## Prerequisites

### System Requirements
- Python 3.10+
- CUDA 11.7+ (for GPU support)
- Docker 20.10+
- Git 2.30+

### Python Environment
```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone repository
git clone https://github.com/username/RxVision.git
cd RxVision

# Install dependencies
poetry install
```

## Development Environment

### IDE Setup
- **VSCode Configuration**
  ```json
  {
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
  ```

### Git Configuration
```bash
# Configure git hooks
pre-commit install

# Setup git config
git config --local core.autocrlf input
git config --local core.eol lf
```

## Docker Setup

### Local Development
```bash
# Build development container
docker compose build dev

# Start development environment
docker compose up dev

# Run tests in container
docker compose run --rm dev pytest
```

### GPU Support
```bash
# Verify GPU access
docker run --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi

# Build with GPU support
docker compose -f docker-compose.gpu.yml build
```

## Data Setup

### Training Data
```bash
# Initialize DVC
dvc init

# Pull training data
dvc pull data/training

# Add new data
dvc add data/new_dataset
dvc push
```

### Model Artifacts
```bash
# Pull pre-trained models
dvc pull models/pretrained

# Track new model
dvc add models/new_model
dvc push
```

## Local Services

### Database
```bash
# Start PostgreSQL
docker compose up -d db

# Run migrations
alembic upgrade head
```

### MLflow
```bash
# Start MLflow server
docker compose up -d mlflow

# Access UI: http://localhost:5000
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=rxvision
```

### Integration Tests
```bash
# Start test environment
docker compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/
```

## Common Issues

### GPU Access
- Verify NVIDIA drivers are installed
- Check docker GPU runtime
- Confirm CUDA compatibility

### Poetry Issues
- Clear poetry cache: `poetry cache clear . --all`
- Recreate virtualenv: `poetry env remove python`
- Update poetry: `poetry self update`

### Docker Issues
- Clean old containers: `docker system prune`
- Reset Docker daemon
- Check disk space

## Next Steps
1. Review [[Code Standards]]
2. Set up [[Local Development]] workflow
3. Read [[Testing Strategy]]

## Related Documentation
- [[Technology Stack]]
- [[Local Development]]
- [[Code Standards]] 