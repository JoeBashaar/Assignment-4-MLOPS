# MLOps — GAN Training Pipeline

This repository contains a Vanilla GAN trained on the Fashion-MNIST dataset, with full MLflow experiment tracking and a GitHub Actions CI/CD pipeline.

## Project Structure

| File / Directory | Description |
|---|---|
| `train_gan.py` | GAN training script with MLflow integration |
| `GAN_Student_A.ipynb` | Jupyter notebook for interactive experimentation |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for reproducible runs |
| `report/` | LaTeX reports for course assignments |
| `.github/workflows/` | GitHub Actions CI pipeline |

## Quick Start

```bash
pip install -r requirements.txt
python train_gan.py
```

## CI Pipeline

The repository includes a GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) that automatically:

1. Checks out the code
2. Sets up Python 3.10
3. Installs dependencies
4. Runs a linter check (flake8)
5. Validates the model environment
6. Uploads `README.md` as a build artifact

The pipeline runs on every push to all branches **except** `main`, and on all pull requests.
