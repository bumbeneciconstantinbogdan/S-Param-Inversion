"""
Data Utilities for Permittivity Prediction

This module provides utilities for data processing, scaling, visualization, and analysis
in the context of microwave permittivity prediction from S-parameters.

Key components:
- TorchScaler classes: Standard and MinMax scalers for PyTorch tensors
- Data generation: Synthetic permittivity data using NRW algorithm
- Data loading: PyTorch DataLoaders with scaling and preprocessing
- Error analysis: Comprehensive error metrics and visualization
- Plotting utilities: Loss curves, error surfaces, R² scatter plots

All functions are designed for reproducibility and GPU compatibility.
"""

import torch
import pandas as pd
import numpy as np
import random
from pathlib import Path
from nrw import nrw_direct
from typing import Optional, Protocol, Union, Any

class TorchScaler(Protocol):
    def fit(self, x: torch.Tensor) -> "TorchScaler": ...

    def transform(self, x: torch.Tensor) -> torch.Tensor: ...

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor: ...

class StandardScalerTorch:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "StandardScalerTorch":
        if x.ndim == 1:
            x = x.unsqueeze(1)
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

class MinMaxScalerTorch:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), eps: float = 1e-8):
        self.feature_range = feature_range
        self.eps = eps
        self.min: Optional[torch.Tensor] = None
        self.max: Optional[torch.Tensor] = None

    def fit(self, x: torch.Tensor) -> "MinMaxScalerTorch":
        if x.ndim == 1:
            x = x.unsqueeze(1)
        self.min = x.amin(dim=0)
        self.max = x.amax(dim=0)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        min_ = self.min.to(device=x.device, dtype=x.dtype)
        max_ = self.max.to(device=x.device, dtype=x.dtype)
        data_range = (max_ - min_).clamp_min(self.eps)
        x_std = (x - min_) / data_range
        fr_min, fr_max = self.feature_range
        return x_std * (fr_max - fr_min) + fr_min

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.min is None or self.max is None:
            raise ValueError("Scaler is not fitted. Call fit() first.")
        fr_min, fr_max = self.feature_range
        min_ = self.min.to(device=x.device, dtype=x.dtype)
        max_ = self.max.to(device=x.device, dtype=x.dtype)
        data_range = (max_ - min_).clamp_min(self.eps)
        x_std = (x - fr_min) / (fr_max - fr_min + self.eps)
        return x_std * data_range + min_

def seed_everything(seed: int) -> None:
    """Set seed for reproducibility across random, numpy, and torch."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Enable deterministic algorithms for reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_nrw_diagnostics(
    data_file: str,
    d: float = 1.5e-3,
    a: float = 22.86e-3,
    frequency: float = 8.2e9,
    thresholds: tuple[float, ...] = (10.0, 1.0),
    save_figures: bool = True,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Run NRW inverse algorithm diagnostics on test data.
    
    Args:
        data_file: Path to CSV file with columns [eps_prime, eps_double_prime, S11_real, S11_imag, S21_real, S21_imag]
        d: Sample thickness (m)
        a: Waveguide width (m)
        frequency: Operating frequency (Hz)
        thresholds: Error thresholds for classification (default: (10.0, 1.0) percent)
        save_figures: Whether to save diagnostic plots
        artifacts_dir: Directory to save artifacts (default: ./artifacts)
    """
    from nrw import nrw_inverse
    
    if artifacts_dir is None:
        artifacts_dir = Path("artifacts")
    else:
        artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(data_file)
    
    # Extract true permittivity (column names: eps_prim, eps_secund)
    eps_prime_true = torch.tensor(df['eps_prim'].values, dtype=torch.float32)
    eps_double_prime_true = torch.tensor(df['eps_secund'].values, dtype=torch.float32)
    
    # Extract S-parameters
    S11_real = torch.tensor(df['S11_real'].values, dtype=torch.float32)
    S11_imag = torch.tensor(df['S11_imag'].values, dtype=torch.float32)
    S21_real = torch.tensor(df['S21_real'].values, dtype=torch.float32)
    S21_imag = torch.tensor(df['S21_imag'].values, dtype=torch.float32)
    
    S11 = S11_real + 1j * S11_imag
    S21 = S21_real + 1j * S21_imag
    
    # Run NRW inverse
    frequencies = torch.full((len(S11),), frequency, dtype=torch.float32)
    eps_r_pred, mu_r_pred = nrw_inverse(d, a, frequencies, S11, S21)
    
    # Extract predicted permittivity (NRW assumes non-magnetic: mu_r = 1)
    eps_prime_pred = eps_r_pred.real
    eps_double_prime_pred = -eps_r_pred.imag  # Convert to positive loss factor
    
    # Prepare targets and predictions in [N, 2] format
    targets = torch.stack([eps_prime_true, eps_double_prime_true], dim=1)
    preds = torch.stack([eps_prime_pred, eps_double_prime_pred], dim=1)
    
    base_name = "nrw_algorithm"
    
    print("\n" + "="*80)
    print("NRW INVERSE ALGORITHM DIAGNOSTICS")
    print("="*80)
    print(f"Test samples: {len(targets)}")
    print(f"Waveguide: a={a*1e3:.2f}mm, d={d*1e3:.2f}mm, f={frequency/1e9:.2f}GHz")
    
    # Generate classification maps for each threshold
    for thr in thresholds:
        print(f"\n--- Threshold: {thr}% ---")
        metrics = analyze_permittivity_error(
            targets,
            preds,
            threshold=float(thr),
            save_figures=bool(save_figures),
            model_name=f"{base_name}_{int(thr)}",
            plot_error_surface=bool(thr == thresholds[0]),
            verbose=True,
        )
    
    # Error histograms
    eps_true_complex = convert_n2_tensor_to_complex(targets)
    eps_pred_complex = convert_n2_tensor_to_complex(preds)
    rel_error_percent, _, _, _, _ = compute_relative_error(eps_true_complex, eps_pred_complex)
    abs_error = np.abs(eps_true_complex - eps_pred_complex)
    
    if save_figures:
        plot_error_histograms(
            rel_error_percent=rel_error_percent,
            abs_error=abs_error,
            save_dir=artifacts_dir,
            base_name=base_name,
        )
        print(f"\nSaved error histograms to {artifacts_dir}")
    
    # R² scores
    targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
    preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
    r2_real = r2_score(targets_np[:, 0], preds_np[:, 0])
    r2_imag = r2_score(targets_np[:, 1], preds_np[:, 1])
    
    print(f"\nR² scores:")
    print(f"  ε' (real):  {r2_real:.5f}")
    print(f"  ε'' (imag): {r2_imag:.5f}")
    
    if save_figures:
        plot_r2_scatter(
            y_true=targets_np,
            y_pred=preds_np,
            r2_real=r2_real,
            r2_imag=r2_imag,
            save_path=artifacts_dir / f"{base_name}_r2_scatter.png",
            title=f"{base_name} — NRW Inverse Predictions",
        )
        print(f"Saved R² scatter plot to {artifacts_dir}")
    
    print("\n" + "="*80)
    print("NRW DIAGNOSTICS COMPLETE")
    print("="*80 + "\n")

def generate_non_magnetic_data(
    d: float = 1.5e-3,
    a: float = 22.86e-3,
    frequency: float = 8.2e9,
    eps_prim_range: tuple[float, float] = (1.0, 200.0),
    eps_secund_range: tuple[float, float] = (0.0, 100.0),
    nopx_train: int = 200,
    nopy_train: int = 100,
    train_ratio: float = 0.8,
    p_val: tuple[float, float] = (0.2, 0.1),
    p_test: tuple[float, float] = (10.0, 5.0),
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training, validation, and test data for non-magnetic materials.
    
    - Training grid: uniform grid with nopx_train × nopy_train points
    - Validation grid: perturbed by p_val offsets
    - Test grid: perturbed by p_test offsets
    
    Args:
        d: sample thickness (m)
        a: waveguide width (m)
        frequency: single frequency (Hz)
        eps_prim_range: (min, max) for real part of permittivity
        eps_secund_range: (min, max) for imaginary part of permittivity
        nopx_train: number of points in x-direction (eps_prim) for training
        nopy_train: number of points in y-direction (eps_secund) for training
        train_ratio: ratio of training data (default 0.8)
        p_val: (px, py) perturbations for validation grid
        p_test: (px, py) perturbations for test grid
        verbose: Whether to print progress information.
        
    Returns:
        train_data, val_data, test_data: each tensor has shape (N, 6)
        columns: [S11_real, S11_imag, S21_real, S21_imag, eps_prim, eps_secund]
    """
    def calculate_dataset_sizes(nopx_train: int, nopy_train: int, train_ratio: float) -> tuple[int, int, int, int, int, int, int]:
        """Calculate the sizes for training, validation, and test datasets.
        
        This function computes grid dimensions for validation and test sets based on
        the training grid size and desired split ratio. The validation and test sets
        are designed to have equal sizes, splitting the remaining data equally.
        
        Args:
            nopx_train: Number of points in x-direction (real permittivity) for training
            nopy_train: Number of points in y-direction (imaginary permittivity) for training
            train_ratio: Fraction of total data to use for training (e.g., 0.8 for 80%)
            
        Returns:
            Tuple containing:
                - nop_train: Total training samples
                - nop_val: Total validation samples
                - nop_test: Total test samples
                - nopx_val: Validation grid x-dimension
                - nopy_val: Validation grid y-dimension
                - nopx_test: Test grid x-dimension
                - nopy_test: Test grid y-dimension
        """
        nop_train = nopx_train * nopy_train
        val_ratio = (1 - train_ratio) / 2
        
        nop_val = int(val_ratio * nop_train / train_ratio)
        nopy_val = int((nop_val / 2) ** 0.5)
        nopx_val = 2 * nopy_val
        nop_val = nopx_val * nopy_val
        
        nopy_test = nopy_val
        nopx_test = nopx_val
        nop_test = nop_val
        
        return nop_train, nop_val, nop_test, nopx_val, nopy_val, nopx_test, nopy_test

    def create_grid_vectors(
        eps_prim_min: float, eps_prim_max: float, eps_secund_min: float, eps_secund_max: float,
        nopx: int, nopy: int, px: float = 0.0, py: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create grid vectors for permittivity values with optional perturbations.
        
        Generates linearly spaced vectors for real and imaginary parts of permittivity.
        Perturbations can be applied to create shifted grids for validation/test sets,
        helping to evaluate model generalization on slightly out-of-distribution data.
        
        Args:
            eps_prim_min: Minimum value for real part of permittivity (ε')
            eps_prim_max: Maximum value for real part of permittivity (ε')
            eps_secund_min: Minimum value for imaginary part of permittivity (ε'')
            eps_secund_max: Maximum value for imaginary part of permittivity (ε'')
            nopx: Number of points in x-direction (real permittivity)
            nopy: Number of points in y-direction (imaginary permittivity)
            px: Perturbation offset for x-direction (default: 0.0)
            py: Perturbation offset for y-direction (default: 0.0)
            
        Returns:
            Tuple of (vect_x, vect_y) containing the grid vectors
        """
        vect_x = torch.linspace(eps_prim_min + px, eps_prim_max - px, nopx)
        vect_y = torch.linspace(eps_secund_min + py, eps_secund_max - py, nopy)
        return vect_x, vect_y

    def generate_for_grid(vect_x: torch.Tensor, vect_y: torch.Tensor, frequency: float, d: float, a: float) -> torch.Tensor:
        """Generate scattering data for a given permittivity grid.
        
        Computes S-parameters (S11 and S21) using the Nicolson-Ross-Weir method for
        each permittivity value in the grid. Creates a meshgrid of complex permittivity
        values and calculates the corresponding scattering parameters.
        
        Args:
            vect_x: Vector of real permittivity values (ε')
            vect_y: Vector of imaginary permittivity values (ε'')
            frequency: Operating frequency in Hz
            d: Sample thickness in meters
            a: Waveguide width in meters
            
        Returns:
            Tensor of shape (n_samples, 6) with columns:
                [S11_real, S11_imag, S21_real, S21_imag, eps_prim, eps_secund]
        
        Note:
            The complex permittivity is defined as ε = ε' - jε'', where ε' is the
            real part (dielectric constant) and ε'' is the imaginary part (loss factor).
        """
        n_samples = len(vect_x) * len(vect_y)
        data = torch.zeros((n_samples, 6), dtype=torch.float64)
        freq_tensor = torch.full((n_samples,), frequency, dtype=torch.float64)
        mu_r_tensor = torch.full((n_samples,), 1+0j, dtype=torch.complex128)

        X, Y = torch.meshgrid(vect_x, vect_y, indexing='ij')
        eps_r_tensor = (X - 1j * Y).reshape(-1).to(torch.complex128)
        
        S11, S21 = nrw_direct(d, a, freq_tensor, eps_r_tensor, mu_r_tensor)

        data[:, 0] = S11.real
        data[:, 1] = S11.imag
        data[:, 2] = S21.real
        data[:, 3] = S21.imag
        data[:, 4] = eps_r_tensor.real
        data[:, 5] = -eps_r_tensor.imag

        return data

    eps_prim_min, eps_prim_max = eps_prim_range
    eps_secund_min, eps_secund_max = eps_secund_range
    p_valx, p_valy = p_val
    p_testx, p_testy = p_test
    
    # Calculate dataset sizes
    nop_train, nop_val, nop_test, nopx_val, nopy_val, nopx_test, nopy_test = calculate_dataset_sizes(
        nopx_train, nopy_train, train_ratio
    )
    
    nop_total = nop_train + nop_val + nop_test

    if verbose:
        print("Training ratios:")
        print(f"Training:   {nop_train/nop_total*100:.2f}% ({nop_train} samples)")
        print(f"Validation: {nop_val/nop_total*100:.2f}% ({nop_val} samples)")
        print(f"Testing:    {nop_test/nop_total*100:.2f}% ({nop_test} samples)")
    
    # Create grids
    train_x, train_y = create_grid_vectors(eps_prim_min, eps_prim_max, eps_secund_min, eps_secund_max, nopx_train, nopy_train)
    val_x, val_y = create_grid_vectors(eps_prim_min, eps_prim_max, eps_secund_min, eps_secund_max, nopx_val, nopy_val, p_valx, p_valy)
    test_x, test_y = create_grid_vectors(eps_prim_min, eps_prim_max, eps_secund_min, eps_secund_max, nopx_test, nopy_test, p_testx, p_testy)
    
    # Generate datasets
    if verbose:
        print("Generating training data...")
    train_data = generate_for_grid(train_x, train_y, frequency, d, a)

    if verbose:
        print("Generating validation data...")
    val_data = generate_for_grid(val_x, val_y, frequency, d, a)

    if verbose:
        print("Generating test data...")
    test_data = generate_for_grid(test_x, test_y, frequency, d, a)
    
    return train_data, val_data, test_data

def save_to_csv(data: torch.Tensor, filepath: str) -> None:
    """Save tensor data to CSV file with proper formatting.
    
    Converts PyTorch tensor to pandas DataFrame and saves as CSV with named columns.
    Creates parent directories if they don't exist.
    
    Args:
        data: Tensor of shape (N, 6) containing scattering parameters and permittivity values
        filepath: Path where the CSV file will be saved
        
    Raises:
        ValueError: If data tensor doesn't have the expected shape
        IOError: If file cannot be written
    """
    if data.shape[1] != 6:
        raise ValueError(f"Expected data with 6 columns, got {data.shape[1]}")
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(data.numpy(), columns=['S11_real', 'S11_imag', 'S21_real', 'S21_imag', 'eps_prim', 'eps_secund'])
    data.to_csv(filepath, index=False)

def create_dataloaders(
    files: tuple[str, str, str] = ("data/data_train.csv", "data/data_val.csv", "data/data_test.csv"),
    batch_sizes: tuple[Union[int, str], Union[int, str], Union[int, str]] = (512, 128, 128),
    num_workers: tuple[int, int, int] = (0, 0, 0),
    transform=None,
    target_transform=None,
    scale_method: str = "standard",
    generator_seed: Optional[int] = None,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
]:
    """
    Create DataLoaders for training, validation, and test datasets.
    
    Args:
        files: Paths to the CSV files for training, validation, and test datasets
        batch_sizes: Batch sizes for training, validation, and test dataloaders
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to samples
        target_transform: Optional transform to apply to labels
        scale_method: Scaling method ('standard', 'minmax', or 'none')
        generator_seed: Optional seed for reproducible data shuffling and worker initialization

    Returns:
        train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    """
    class EpsilonDataSet(torch.utils.data.Dataset):
        def __init__(
            self,
            file,
            transform=None,
            target_transform=None,
            x_scaler: Optional[TorchScaler] = None,
            y_scaler: Optional[TorchScaler] = None,
        ):
            self.data = pd.read_csv(file)
            self.samples = self.data[['S11_real', 'S11_imag', 'S21_real', 'S21_imag']].values
            self.labels = self.data[['eps_prim', 'eps_secund']].values
            self.transform = transform
            self.target_transform = target_transform
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = torch.tensor(self.samples[idx], dtype=torch.float32)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.x_scaler is not None:
                sample = self.x_scaler.transform(sample)
            if self.y_scaler is not None:
                label = self.y_scaler.transform(label)

            if self.transform:
                sample = self.transform(sample)
            if self.target_transform:
                label = self.target_transform(label)
            return sample, label

    scale_data = scale_method.lower() != "none"

    if scale_data:
        train_df = pd.read_csv(files[0])
        x_train = torch.tensor(
            train_df[['S11_real', 'S11_imag', 'S21_real', 'S21_imag']].values,
            dtype=torch.float32,
        )
        y_train = torch.tensor(
            train_df[['eps_prim', 'eps_secund']].values,
            dtype=torch.float32,
        )

        method = (scale_method or "standard").lower()
        if method == "standard":
            x_scaler = StandardScalerTorch().fit(x_train)
            y_scaler = StandardScalerTorch().fit(y_train)
        elif method == "minmax":
            x_scaler = MinMaxScalerTorch().fit(x_train)
            y_scaler = MinMaxScalerTorch().fit(y_train)
        elif method == "none":
            x_scaler = None
            y_scaler = None
        else:
            raise ValueError(f"Unknown scale_method: {scale_method}. Use 'standard', 'minmax', or 'none'.")
    else:
        x_scaler = None
        y_scaler = None

    train_dataset = EpsilonDataSet(files[0], transform=transform, target_transform=target_transform, x_scaler=x_scaler, y_scaler=y_scaler)
    val_dataset = EpsilonDataSet(files[1], transform=transform, target_transform=target_transform, x_scaler=x_scaler, y_scaler=y_scaler)
    test_dataset = EpsilonDataSet(files[2], transform=transform, target_transform=target_transform, x_scaler=x_scaler, y_scaler=y_scaler)

    # Create generator for reproducible shuffling
    g = None
    if generator_seed is not None:
        g = torch.Generator()
        g.manual_seed(generator_seed)
    
    def worker_init_fn(worker_id):
        """Initialize each dataloader worker with a unique but deterministic seed."""
        if generator_seed is not None:
            worker_seed = generator_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_sizes[0] if batch_sizes[0] != "ALL" else len(train_dataset),
        shuffle=True,
        num_workers=num_workers[0],
        generator=g,
        worker_init_fn=worker_init_fn if generator_seed is not None else None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_sizes[1] if batch_sizes[1] != "ALL" else len(val_dataset),
        shuffle=False,
        num_workers=num_workers[1],
        worker_init_fn=worker_init_fn if generator_seed is not None else None
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_sizes[2] if batch_sizes[2] != "ALL" else len(test_dataset),
        shuffle=False,
        num_workers=num_workers[2],
        worker_init_fn=worker_init_fn if generator_seed is not None else None
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def compute_relative_error(z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, int, float, float, float]:
    """
    Compute relative error for permittivity predictions.
    
    Args:
        z1: Array of true complex permittivity values
        z2: Array of predicted complex permittivity values
        
    Returns:
        Tuple of (rel_error, n_samples, mean_error, max_error, min_error)
    """
    n_samples = len(z1)
    
    # Calculate relative error (in percent)
    rel_error = np.abs(z1 - z2) / (np.abs(z1) + 1e-16) * 100
    
    mean_error = float(np.mean(rel_error))
    max_error = float(np.max(rel_error))
    min_error = float(np.min(rel_error))
    
    return rel_error, n_samples, mean_error, max_error, min_error

def compute_ok_classification(rel_error: np.ndarray, threshold: float, n_samples: int) -> tuple[np.ndarray, int, float]:
    """
    Compute OK/Not OK classification metrics.
    
    Args:
        rel_error: Array of relative errors in percent
        threshold: Error threshold for OK/Not OK classification (percent)
        n_samples: Number of samples
        
    Returns:
        Tuple of (ok_mask, ok_count, ok_percent)
    """
    ok_mask = rel_error <= threshold
    ok_count = np.sum(ok_mask)
    ok_percent = (ok_count / n_samples) * 100
    return ok_mask, ok_count, ok_percent

def convert_n2_tensor_to_complex(eps_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor of shape (N, 2) representing real and imaginary parts
    into a complex numpy array of shape (N,).
    
    Args:
        eps_tensor: Tensor of shape (N, 2) where column 0 is real part and column 1 is imaginary part
        
    Returns:
        Numpy array of complex permittivity values of shape (N,)
    """
    eps_tensor_np = eps_tensor.numpy()
    return eps_tensor_np[:, 0] - 1j * eps_tensor_np[:, 1]

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 score (coefficient of determination) for a 1D target."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot

def plot_loss_history(logs: dict, save_path: Union[str, Path], title: str, *, log_y: bool = True) -> None:
    """Plot training/validation loss vs epoch.

    Expects Trainer-style logs: {'train_loss': [...], 'val_loss': [...]}.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[data_utils] Matplotlib not available; skipping loss history plot ({e}).")
        return

    train_loss = list(logs.get("train_loss", []))
    val_loss = list(logs.get("val_loss", []))
    if len(train_loss) == 0 and len(val_loss) == 0:
        return

    if log_y:
        eps = 1e-12
        train_loss = [max(float(x), eps) for x in train_loss]
        val_loss = [max(float(x), eps) for x in val_loss]

    plt.figure(figsize=(7, 4))
    if len(train_loss) > 0:
        plt.plot(train_loss, label="train")
    if len(val_loss) > 0:
        plt.plot(val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    if log_y:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_error_histograms(
    *,
    rel_error_percent: np.ndarray,
    abs_error: np.ndarray,
    save_dir: Union[str, Path],
    base_name: str,
) -> None:
    """Save relative-error (%) and absolute-error (magnitude) histograms.

    Note: abs_error should be a magnitude (e.g. |ε_true - ε_pred|), therefore always >= 0.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[data_utils] Matplotlib not available; skipping histograms ({e}).")
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rel_error_percent = np.asarray(rel_error_percent, dtype=np.float64)
    abs_error = np.asarray(abs_error, dtype=np.float64)

    plt.figure(figsize=(7, 4))
    plt.hist(rel_error_percent[np.isfinite(rel_error_percent)], bins=60, color="#1f77b4", alpha=0.85)
    plt.xlabel("relative error [%]")
    plt.ylabel("count")
    plt.title(f"{base_name} — Relative error distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"{base_name}_rel_error_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(abs_error[np.isfinite(abs_error)], bins=60, color="#ff7f0e", alpha=0.85)
    plt.xlabel(r"absolute complex error magnitude |Δε| [unitless]")
    plt.ylabel("count")
    plt.title(f"{base_name} — Absolute error distribution")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"{base_name}_abs_error_hist.png", dpi=200)
    plt.close()

def plot_r2_scatter(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2_real: float,
    r2_imag: float,
    save_path: Union[str, Path],
    title: str,
) -> None:
    """Scatter of pred vs true for eps' and eps'' with R^2 annotation."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[data_utils] Matplotlib not available; skipping R2 plot ({e}).")
        return

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        return
    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    parts = [(0, "ε' (real)", r2_real), (1, "ε'' (imag)", r2_imag)]
    for ax, (idx, label, r2) in zip(axes, parts):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        ax.scatter(yt, yp, s=10, alpha=0.55)
        finite = np.isfinite(yt) & np.isfinite(yp)
        if np.any(finite):
            vmin = float(np.min([np.min(yt[finite]), np.min(yp[finite])]))
            vmax = float(np.max([np.max(yt[finite]), np.max(yp[finite])]))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
            ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1.0, color="black", alpha=0.7)
            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)

        ax.set_xlabel(f"true {label}")
        ax.set_ylabel(f"pred {label}")
        ax.set_title(f"R² = {r2:.5f}")
        ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def analyze_permittivity_error(
    eps_true: torch.Tensor,
    eps_pred: torch.Tensor,
    threshold: float = 10.0,
    model_name: str = "model",
    save_figures: bool = False,
    plot_error_surface: bool = False,
    verbose: bool = True
) -> dict:
    """
    Analyzes and visualizes the error between true and predicted permittivity samples.
    
    Computes numerical metrics and optionally creates visualizations.
    
    Args:
        eps_true: Tensor of true permittivity values [N, 2] where columns are [real, imag]
        eps_pred: Tensor of predicted permittivity values [N, 2]
        threshold: Error threshold for OK/Not OK classification (percent, default: 10.0)
        save_figures: If True, save figures to disk and create plots
        model_name: Name prefix for saved files (default: "model")
        plot_error_surface: If True, plot the 3D error surface
        
    Returns:
        Dictionary containing analysis metrics:
            - n_samples: Number of samples analyzed
            - ok_count: Number of samples within threshold
            - ok_percent: Percentage of samples within threshold
            - mean_error: Mean relative error
            - max_error: Maximum relative error
            - min_error: Minimum relative error
            - threshold: The threshold used
    """
    def compute_permittivity_error_metrics(
        eps_true: torch.Tensor,
        eps_pred: torch.Tensor,
        threshold: float = 10.0,
        compute_grid: bool = False
    ) -> dict:
   
        
        # Convert to complex numbers
        eps_true_complex = convert_n2_tensor_to_complex(eps_true)
        eps_pred_complex = convert_n2_tensor_to_complex(eps_pred)
        
        rel_error, n_samples, mean_error, max_error, min_error = compute_relative_error(eps_true_complex, eps_pred_complex)
        
        # --- Compute OK/Not OK classification ---
        ok_mask, ok_count, ok_percent = compute_ok_classification(rel_error, threshold, n_samples)
        
        result = {
            'rel_error': rel_error,
            'ok_mask': ok_mask,
            'n_samples': n_samples,
            'ok_count': ok_count,
            'ok_percent': ok_percent,
            'mean_error': mean_error,
            'max_error': max_error,
            'min_error': min_error,
            'threshold': threshold
        }
        
        if compute_grid:
            # --- Detect grid structure ---
            # Use positive ε' (real part) and ε'' (imaginary part, made positive)
            eps_prime = eps_true_complex.real  # ε'
            eps_double_prime = -eps_true_complex.imag  # ε'' (positive)
            
            uniq_eps_prime = np.unique(np.round(eps_prime, decimals=6))
            uniq_eps_double_prime = np.unique(np.round(eps_double_prime, decimals=6))
            nx, ny = len(uniq_eps_prime), len(uniq_eps_double_prime)
            
            is_grid = (nx * ny == n_samples and ny > 1)
            
            error_grid = None
            if is_grid:
                # Create mapping for grid reconstruction using positive ε' and ε''
                error_grid = np.full((nx, ny), np.nan)
                
                # Vectorized grid reconstruction for better performance
                eps_prime_rounded = np.round(eps_prime, 6)
                eps_double_prime_rounded = np.round(eps_double_prime, 6)
                
                ri = np.searchsorted(uniq_eps_prime, eps_prime_rounded)
                ii = np.searchsorted(uniq_eps_double_prime, eps_double_prime_rounded)
                
                # Ensure exact matches
                valid_ri = (ri < nx) & (uniq_eps_prime[ri] == eps_prime_rounded)
                valid_ii = (ii < ny) & (uniq_eps_double_prime[ii] == eps_double_prime_rounded)
                valid = valid_ri & valid_ii
                
                error_grid[ri[valid], ii[valid]] = rel_error[valid]
            
            result.update({
                'is_grid': is_grid,
                'error_grid': error_grid,
                'uniq_eps_prime': uniq_eps_prime if is_grid else None,
                'uniq_eps_double_prime': uniq_eps_double_prime if is_grid else None,
                'eps_prime': eps_prime,
                'eps_double_prime': eps_double_prime
            })
        
        return result

    def plot_permittivity_error(
    metrics: dict,
    save_figures: bool = False,
    model_name: str = "model",
    plot_error_surface: bool = False
) -> None:
        """
        Plots permittivity error visualizations.
        
        Args:
            metrics: Dictionary from compute_permittivity_error_metrics
            save_figures: If True, save figures to disk
            model_name: Name prefix for saved files
            plot_error_surface: If True, plot the 3D error surface
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        rel_error = metrics['rel_error']
        is_grid = metrics['is_grid']
        error_grid = metrics['error_grid']
        uniq_eps_prime = metrics['uniq_eps_prime']
        uniq_eps_double_prime = metrics['uniq_eps_double_prime']
        eps_prime = metrics['eps_prime']
        eps_double_prime = metrics['eps_double_prime']
        ok_mask = metrics['ok_mask']
        n_samples = metrics['n_samples']
        ok_count = metrics['ok_count']
        ok_percent = metrics['ok_percent']
        threshold = metrics['threshold']
        
        # --- Figure 1: 3D Error Surface (only for grid data) ---
        if is_grid and plot_error_surface:
            from matplotlib.colors import LogNorm
            
            X, Y = np.meshgrid(uniq_eps_prime, uniq_eps_double_prime, indexing='ij')
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Compute a safe floor (smallest positive finite value) and set
            # plotting limits to full decades so each decade occupies equal
            # vertical space after mapping Z->log10(Z).
            finite_mask = np.isfinite(error_grid) & (error_grid > 0)
            if np.any(finite_mask):
                min_pos = float(np.nanmin(error_grid[finite_mask]))
            else:
                min_pos = 1e-6

            # Choose lower decade (at most -6) and upper decade (1 => 10)
            decade_min = max(int(np.floor(np.log10(min_pos))), -6)
            decade_max = 1

            vmin = 10.0 ** decade_min
            vmax = 10.0 ** decade_max

            error_grid_clipped = np.clip(error_grid, vmin, vmax)
            Z_plot = np.log10(error_grid_clipped)

            # Use LogNorm for color mapping but pass facecolors computed from
            # the original (non-log) error grid so colorbar shows actual values.
            from matplotlib import cm
            cmap = cm.get_cmap('viridis')
            norm = LogNorm(vmin=vmin, vmax=vmax)
            facecolors = cmap(norm(error_grid_clipped))

            surf = ax.plot_surface(X, Y, Z_plot, facecolors=facecolors, linewidth=0, antialiased=False)
            ax.set_xlabel("ε'", fontsize=10)
            ax.set_ylabel("ε''", fontsize=10)
            ax.set_zlabel('Relative Error (%) [log scale]', fontsize=10)

            # z axis is in log10 units -> ticks at integer decades
            z_ticks = list(range(decade_min, decade_max + 1))
            ax.set_zlim(z_ticks[0], z_ticks[-1])
            ax.set_zticks(z_ticks)
            ax.set_zticklabels([f"1e{d}" for d in z_ticks])
            ax.set_title('3D Error Surface (log scale)', fontsize=12, fontweight='bold')

            # colorbar built from a ScalarMappable using same norm/cmap
            from matplotlib.cm import ScalarMappable
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6, label='Relative Error (%) [log scale]')
            cbar.set_ticks([10.0 ** d for d in z_ticks])
            cbar.set_ticklabels([f"1e{d}" for d in z_ticks])
            
            plt.tight_layout()
            if save_figures:
                import os
                save_dir = os.path.join(os.path.dirname(__file__), "artifacts")
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{model_name}_error_surface.png"), dpi=300, bbox_inches='tight')
                print(f"Saved: {os.path.join(save_dir, f'{model_name}_error_surface.png')}")
        
        # --- Figure 2: OK/Not OK Classification Map ---
        fig, ax = plt.subplots(figsize=(10, 7))
        
        if is_grid:
            nx, ny = len(uniq_eps_prime), len(uniq_eps_double_prime)
            ok_grid = np.full((nx, ny), np.nan)
            eps_prime_idx = {round(v, 6): i for i, v in enumerate(uniq_eps_prime)}
            eps_double_prime_idx = {round(v, 6): i for i, v in enumerate(uniq_eps_double_prime)}
            
            for i in range(n_samples):
                ri = eps_prime_idx.get(round(eps_prime[i], 6))
                ii = eps_double_prime_idx.get(round(eps_double_prime[i], 6))
                if ri is not None and ii is not None:
                    ok_grid[ri, ii] = 1 if ok_mask[i] else 0
            
            # Calculate grid spacing for enhanced grid lines
            eps_prime_spacing = (uniq_eps_prime.max() - uniq_eps_prime.min()) / (nx - 1) if nx > 1 else 1
            eps_double_prime_spacing = (uniq_eps_double_prime.max() - uniq_eps_double_prime.min()) / (ny - 1) if ny > 1 else 1
            
            im = ax.imshow(
                ok_grid.T,
                extent=[uniq_eps_prime.min() - eps_prime_spacing/2, 
                        uniq_eps_prime.max() + eps_prime_spacing/2,
                        uniq_eps_double_prime.min() - eps_double_prime_spacing/2, 
                        uniq_eps_double_prime.max() + eps_double_prime_spacing/2],
                origin='lower',
                aspect='auto',
                cmap='RdYlGn',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            
            # Add grid lines to show each square element
            for i in range(nx + 1):
                x_pos = uniq_eps_prime.min() - eps_prime_spacing/2 + i * eps_prime_spacing
                ax.axvline(x=x_pos, color='black', linewidth=0.5, alpha=0.5)
            for j in range(ny + 1):
                y_pos = uniq_eps_double_prime.min() - eps_double_prime_spacing/2 + j * eps_double_prime_spacing
                ax.axhline(y=y_pos, color='black', linewidth=0.5, alpha=0.5)
            
            ax.set_xlabel("ε'", fontsize=11)
            ax.set_ylabel("ε''", fontsize=11)
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Not OK', 'OK'])
        else:
            # For non-grid data, use scatter plot
            sample_idx = np.arange(n_samples)
            colors = ['red' if not ok else 'green' for ok in ok_mask]
            ax.scatter(sample_idx, rel_error, c=colors, s=20, alpha=0.6)
            ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold}%)')
            ax.set_xlabel('Sample Index', fontsize=11)
            ax.set_ylabel('Relative Error (%)', fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        ax.set_title(f'Classification Map: {ok_count}/{n_samples} OK ({ok_percent:.1f}%)', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_figures:
            import os
            save_dir = os.path.join(os.path.dirname(__file__), "artifacts")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"{model_name}_classification_map.png"), dpi=300, bbox_inches='tight')
            print(f"Saved: {os.path.join(save_dir, f'{model_name}_classification_map.png')}")
        # plt.show()  # Removed to avoid displaying plots

    # Compute numerical metrics
    compute_grid = save_figures or plot_error_surface
    metrics = compute_permittivity_error_metrics(eps_true, eps_pred, threshold, compute_grid)
    
    # Print summary statistics
    if verbose:
        n_samples = metrics['n_samples']
        ok_count = metrics['ok_count']
        ok_percent = metrics['ok_percent']
        is_grid = metrics.get('is_grid', False)
        uniq_eps_prime = metrics.get('uniq_eps_prime')
        uniq_eps_double_prime = metrics.get('uniq_eps_double_prime')
        
        print(f"\n{'='*60}")
        print(f"Permittivity Error Analysis Summary")
        print(f"{'='*60}")
        print(f"Total samples:     {n_samples}")
        print(f"Grid structure:    {len(uniq_eps_prime)} x {len(uniq_eps_double_prime)}" if is_grid and uniq_eps_prime is not None and uniq_eps_double_prime is not None else f"Grid structure:    Linear ({n_samples} points)")
        print(f"Error threshold:   {threshold}%")
        print(f"Samples within threshold: {ok_count}/{n_samples} ({ok_percent:.2f}%)")
        print(f"Mean error:        {metrics['mean_error']:.4f}%")
        print(f"Max error:         {metrics['max_error']:.4f}%")
        print(f"Min error:         {metrics['min_error']:.4f}%")
        print(f"{'='*60}\n")
    
    # Create plots only if requested
    if save_figures or plot_error_surface:
        plot_permittivity_error(metrics, save_figures, model_name, plot_error_surface)
    
    # Return only the metrics
    return {
        'n_samples': metrics['n_samples'],
        'ok_count': metrics['ok_count'],
        'ok_percent': metrics['ok_percent'],
        'mean_error': metrics['mean_error'],
        'max_error': metrics['max_error'],
        'min_error': metrics['min_error'],
        'threshold': metrics['threshold']
    }