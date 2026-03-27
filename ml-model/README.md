# Cafe Supply Forecasting

This project provides tools for managing and exporting Python environments for the cafe supply forecasting system.

## Environment Setup

This project uses Conda for environment management with Python 3.13. The environment includes essential packages for data processing, validation, and CLI operations.

## Files Overview

### [`environment.yml`](environment.yml)
Conda environment specification file that includes:
- Python 3.13.9
- All conda-managed packages with specific versions
- Channel configuration (defaults)
- Complete dependency tree for reproducible environments

### [`requirements.txt`](requirements.txt)
Pip requirements file containing:
- All pip-installed packages
- Development and build dependencies
- Package references with exact versions for reproducibility

### [`export-env.sh`](export-env.sh)
Automated environment export script that:
- Detects the current Conda environment name
- Exports the full Conda environment to `environment.yml`
- Exports pip packages to `requirements.txt`
- Cleans up the environment.yml file by removing system prefix paths

## Quick Start

### Option 1: Using the Export Script (Recommended)

If you want to replicate the current environment:

```bash
# Make the script executable
chmod +x export-env.sh

# Run the export script
./export-env.sh
```

This will generate both [`environment.yml`](environment.yml) and [`requirements.txt`](requirements.txt) files based on the current environment.

### Option 2: Creating Environment from Scratch

1. **Create new conda environment from environment.yml:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate base
   ```

3. **Or create environment manually:**
   ```bash
   conda create -n cafe-supply-forecasting python=3.13
   conda activate cafe-supply-forecasting
   pip install -r requirements.txt
   ```

## Environment Management

### Exporting Environment Changes

When you install new packages or update versions, run the export script to update the environment files:

```bash
./export-env.sh
```

### Updating Environment

To update existing packages:

```bash
# Update conda packages
conda update --all

# Update pip packages
pip install --upgrade -r requirements.txt

# Export updated environment
./export-env.sh
```

### Removing Environment

```bash
conda env remove -n base
```

## Development Workflow

1. **Activate environment:**
   ```bash
   conda activate base
   ```

2. **Install new dependencies:**
   ```bash
   # For conda packages
   conda install package-name
   
   # For pip packages
   pip install package-name
   ```

3. **Export updated environment:**
   ```bash
   ./export-env.sh
   ```

4. **Commit changes:**
   ```bash
   git add environment.yml requirements.txt
   git commit -m "Update environment dependencies"
   ```

## Troubleshooting

### Common Issues

1. **Environment not found:**
   ```bash
   conda info --envs
   # Check if the environment exists
   ```

2. **Package conflicts:**
   ```bash
   conda env update -f environment.yml
   # This will try to resolve conflicts
   ```

3. **Permission issues with export script:**
   ```bash
   chmod +x export-env.sh
   ```

### Clean Installation

If you encounter issues, try a clean installation:

```bash
# Remove existing environment
conda env remove -n base

# Create fresh environment
conda env create -f environment.yml

# Activate
conda activate base
```

## Project Structure

```
cafe-supply-forecasting/
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements file
├── export-env.sh                 # Environment export script
├── data/                         # Data directory
│   ├── predictions/              # Model predictions
│   │   └── example.csv
│   └── raw/                      # Raw data files
│   └── processed/                # Processed data files
├── models/                       # Trained model files
│   └── example.pkl
├── notebooks/                    # Jupyter notebooks
│   └── example.ipynb
├── scripts/                      # Utility scripts
│   ├── forecast.py               # Forecasting script
│   └── train_models.py           # Model training script
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   └── __init__.py
│   ├── evaluation/               # Model evaluation modules
│   │   └── __init__.py
│   ├── models/                   # Model definitions
│   │   └── __init__.py
│   └── utils/                    # Utility functions
│       └── __init__.py
└── tests/                        # Unit Test files
    └── __init__.py
```
