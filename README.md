
# Neural Network-Based Inverse Permittivity Characterization


![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-ee4c2c)
![Optuna](https://img.shields.io/badge/Optuna-HPO-406599)
![License](https://img.shields.io/badge/license-MIT-green)

This repository implements a compact **Multilayer Perceptron (MLP)** to solve the inverse permittivity problem for **non-magnetic materials** (μ_r = 1) using waveguide S-parameter measurements.

The network maps measured S-parameters (S11, S21) directly to the complex relative permittivity ε_r = ε' - jε'', providing a robust, noise-resilient alternative to the classical Nicolson–Ross–Weir (NRW) closed-form inversion method.

## Features

* **Non-Magnetic Assumption:** Optimized specifically for materials where μ_r ≈ 1.
* **Physics-Informed Data:** Uses an analytical TE10 forward model to generate synthetic training, validation, and test datasets.
* **Robust Architecture:** Single-hidden-layer MLP designed for low latency and high accuracy.
* **Advanced HPO:** Automated hyperparameter optimization using **Optuna** with the **NSGA-II** multi-objective algorithm (optimizing for accuracy and model size).
* **Benchmarking:** Includes tools to compare Neural Network performance against the classical NRW method.

## Repository Structure

```text
.
├── artifacts/               # Stores study database (SQLite) and generated figures
├── data/                    # Generated CSV datasets (train/val/test)
├── data_utils.py            # Forward model, dataset generation, scaling, and plotting
├── hyper_param_optimize.py  # Optuna wrapper (objectives, constraints, visualization)
├── main.py                  # Entry point: orchestrates data gen, HPO, and diagnostics
├── mlp.py                   # PyTorch model definition, training loop, and evaluation
├── nrw.py                   # Classical Nicolson–Ross–Weir implementation (baseline)
└── requirements.txt         # Project dependencies

```

## Installation

Python 3.8 or later is required. It is recommended to use a virtual environment.

```bash
# Create and activate virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

## Quickstart & Usage

The `main.py` script serves as the central command center. You can control its behavior using environment variables.

### 1. Generate Data & Run Diagnostics

Regenerate synthetic datasets and run a summary of the current model/study state without starting new training trials.

```bash
REGENERATE_DATA=True SUMMARY_ONLY=True python3 main.py

```

### 2. Run Hyperparameter Optimization (HPO)

Run the Optuna optimization loop.

```bash
# Run 100 trials on CPU
SUMMARY_ONLY=False OPTUNA_N_TRIALS=100 python3 main.py

# Run on GPU (if available) with 200 trials
DEVICE=cuda SUMMARY_ONLY=False OPTUNA_N_TRIALS=200 python3 main.py

```

### 3. Reproduce & Visualize Results

Once a study exists in `artifacts/`, you can generate performance plots and metrics.

```bash
SUMMARY_ONLY=True python3 main.py

```

## Configuration

You can override default behaviors by exporting the following environment variables:

| Variable | Default | Options | Description |
| --- | --- | --- | --- |
| `DEVICE` | `cpu` | `cpu`, `cuda`, `mps` | Compute device for PyTorch. |
| `SCALE_METHOD` | `standard` | `standard`, `minmax`, `none` | Input feature scaling strategy. |
| `SUMMARY_ONLY` | `True` | `True`, `False` | If `True`, skips training and only runs diagnostics. |
| `REGENERATE_DATA` | `True` | `True`, `False` | Forces regeneration of synthetic `.csv` datasets. |
| `OPTUNA_N_TRIALS` | `10000` | Integer | Number of HPO trials to run. |
| `OPTUNA_N_JOBS` | `1` | Integer | Number of parallel Optuna jobs. |
| `OPTUNA_PARALLEL_MODE` | `process` | `process`, `thread` | Parallelization backend. |

## Important Assumptions

**Non-Magnetic Materials Only:**
All datasets and experiments in this repository are created under the assumption that μ_r = 1. If you intend to characterize magnetic materials (μ_r ≠ 1), the forward model in `data_utils.py` and the dataset generation logic must be adapted.

## Results Summary

The model is evaluated using **Relative Local Error** and **Strict Accuracy** (OK@τ).

* **OK@1%:** Percentage of test samples with <1% relative error.
* **OK@10%:** Percentage of test samples with <10% relative error.

**Current Best Performance (Synthetic Data):**

* **OK@1%:** ~99.4%
* **Comparison:** The MLP significantly outperforms the NRW baseline, particularly in avoiding singularity instabilities common in analytical inversion.

## Notes & Deployment

* **Real World Data:** This project uses synthetic data. Before deploying in experimental settings, validate with measured S-parameters and consider adding noise models during training.
* **Training Your Own Model:**
1. Run the HPO to populate `artifacts/study.db`.
2. Identify the best trial parameters.
3. Retrain the model on the full dataset.
4. Save using `torch.save()`.


* **Inference:** Ensure that input S-parameters are scaled using the same statistics (mean/std or min/max) used during training.

## License & Citation

This project is open-source. If you use this code or methodology in your research, please cite the associated paper (once will be published):

```bibtex
@article{}
```

For questions, issues, or contributions, please open an Issue or Pull Request on GitHub.
