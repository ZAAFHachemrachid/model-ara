# Project Structure

```
model-ara/
├── main.py              # Application entry point
├── pyproject.toml       # Project configuration and dependencies
├── .python-version      # Python version specification (3.12)
├── README.md            # Project documentation
└── dataset/             # Training and evaluation data
    ├── train.csv        # Training dataset (~2800 samples)
    ├── test.csv         # Test dataset (~600 samples)
    └── validation.csv   # Validation dataset (~600 samples)
```

## Key Files

### main.py
Entry point for the application. Currently a placeholder.

### dataset/
Contains CSV files with Arabic news articles for classification:
- Pre-split into train/test/validation sets
- Each file has columns: title, text, subject, date, validity
- Binary classification target: validity (0=fake, 1=real)

## Conventions
- Keep ML model code separate from data loading utilities
- Use the existing train/test/validation split for experiments
- Store model artifacts outside the dataset directory
