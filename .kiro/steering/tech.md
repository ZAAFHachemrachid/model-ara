# Tech Stack

## Language & Runtime
- Python 3.12

## Build System
- Uses `pyproject.toml` for project configuration (PEP 517/518 compliant)
- No external dependencies currently defined

## Project Configuration
```toml
[project]
name = "model-ara"
version = "0.1.0"
requires-python = ">=3.12"
```

## Common Commands

### Run the application
```bash
python main.py
```

### Install dependencies (when added)
```bash
UV pip install -e .
```

## Potential Libraries (for ML development)
When implementing the classification model, consider:
- `pandas` - Data manipulation
- `scikit-learn` - ML algorithms
- `transformers` - Arabic NLP models (e.g., AraBERT)
- `torch` or `tensorflow` - Deep learning frameworks
