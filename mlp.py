"""
MLP Model and Training Utilities

This module provides PyTorch implementations for MLP regression models, optimizers,
activation functions, and training infrastructure. Designed for hyperparameter
optimization workflows with support for various regularization techniques,
learning rate scheduling, and early stopping.

Key components:
- MLPRegressor: Configurable neural network for regression tasks
- Trainer: Training loop with early stopping and validation monitoring
- Factory functions: Optimizers, activations, and parameter builders
- Utilities: Prediction extraction and model reconstruction from parameters

Supports GPU training, gradient clipping, input noise injection, and multiple loss functions.
"""

import time
import copy
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, Optional, Union

def make_activation(name: str) -> nn.Module:
    """Create a PyTorch activation module from a string."""
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unknown activation: {name}")

def make_optimizer(name: str, parameters: Iterable[torch.nn.Parameter], lr: float, **kwargs):
    """Create a PyTorch optimizer from a string name.

    Args:
        name: Optimizer name ('adam', 'adamw', 'rmsprop', 'sgd')
        parameters: Model parameters to optimize
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters:
            - adam: betas (tuple, default (0.9, 0.999)), eps (float, default 1e-8), weight_decay (float, default 0)
            - adamw: betas (tuple, default (0.9, 0.999)), eps (float, default 1e-8), weight_decay (float, default 0.01)
            - rmsprop: alpha (float, default 0.99), eps (float, default 1e-8), weight_decay (float, default 0), momentum (float, default 0)
            - sgd: momentum (float, default 0), dampening (float, default 0), weight_decay (float, default 0), nesterov (bool, default False)
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, **kwargs)
    if name == "lbfgs":
        # LBFGS ignores minibatch stochasticity; best used with full batch.
        # torch.optim.LBFGS does NOT support weight_decay.
        kwargs.pop("weight_decay", None)
        return torch.optim.LBFGS(parameters, lr=lr, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")

class MLPRegressor(nn.Module):
    """Simple configurable MLP regressor for regression tasks.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        hidden_size: Size of the single hidden layer.
        activation: Activation function name ('tanh', 'relu', 'gelu', 'sigmoid', 'leaky_relu', 'elu', 'selu').
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        activation: str,
        *,
        dropout_p: float = 0.0,
        norm: str = "none",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.activation_name = activation

        layers = []
        in_features = input_size
        act = make_activation(activation)

        layers.append(nn.Linear(in_features, hidden_size))
        norm_name = str(norm).lower()
        if norm_name == "batchnorm":
            layers.append(nn.BatchNorm1d(hidden_size))
        elif norm_name == "layernorm":
            layers.append(nn.LayerNorm(hidden_size))
        elif norm_name in {"none", ""}:
            pass
        else:
            raise ValueError(f"Unknown norm: {norm}")

        layers.append(act)
        if float(dropout_p) > 0:
            layers.append(nn.Dropout(p=float(dropout_p)))
        in_features = hidden_size

        self.feature = nn.Sequential(*layers)
        self.out = nn.Linear(in_features, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.out(self.feature(x))

class Trainer:
    """A simple trainer for PyTorch neural network models with early stopping.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer_name: Name of the optimizer ('adam', 'adamw', 'rmsprop', 'sgd').
        lr: Learning rate.
        device: Device to run on ('cpu' or 'cuda').
        patience: Number of epochs to wait for improvement before early stopping.
        warmup: Number of epochs to train before checking for early stopping.
        min_delta: Minimum change in validation loss to qualify as an improvement.
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer_name: str = 'adam',
        lr: float = 1e-3,
        device='cpu',
        patience: int = 15,
        warmup: int = 5,
        min_delta: float = 0.0,
        loss_name: str = "mse",
        scheduler_name: str = "none",
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip_norm: Optional[float] = None,
        input_noise_std: float = 0.0,
        **optimizer_kwargs
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.device = device
        self.patience = patience
        self.warmup = warmup
        self.min_delta = min_delta
        self.loss_name = loss_name
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.grad_clip_norm = grad_clip_norm
        self.input_noise_std = float(input_noise_std)
        self.optimizer_kwargs = optimizer_kwargs

    def train(self, model: nn.Module, max_epochs: int) -> dict:
        """Train the model for max_epochs with early stopping."""
        model = model.to(self.device)
        optimizer = make_optimizer(self.optimizer_name, model.parameters(), self.lr, **self.optimizer_kwargs)

        def _criterion(pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
            name = str(self.loss_name).lower()
            if name in {"mse", "l2"}:
                return torch.nn.functional.mse_loss(pred, yb)
            if name in {"smooth_l1", "huber"}:
                return torch.nn.functional.smooth_l1_loss(pred, yb)
            if name in {"eps_rel_mse", "rel_mse"}:
                # Relative error in the *physical* (unscaled) epsilon space.
                y_scaler = getattr(getattr(self.train_loader, "dataset", None), "y_scaler", None)
                if y_scaler is not None:
                    pred_u = y_scaler.inverse_transform(pred)
                    yb_u = y_scaler.inverse_transform(yb)
                else:
                    pred_u, yb_u = pred, yb

                eps_true = yb_u[:, 0].to(torch.complex64) - 1j * yb_u[:, 1].to(torch.complex64)
                eps_pred = pred_u[:, 0].to(torch.complex64) - 1j * pred_u[:, 1].to(torch.complex64)
                denom = torch.abs(eps_true).clamp_min(1e-6)
                rel = torch.abs(eps_pred - eps_true) / denom
                return torch.mean(rel * rel)
            raise ValueError(f"Unknown loss_name: {self.loss_name}")

        scheduler = None
        sched_name = str(self.scheduler_name).lower()
        if sched_name not in {"none", "", "null"}:
            if sched_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=int(max_epochs),
                    **self.scheduler_kwargs,
                )
            elif sched_name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_kwargs)
            elif sched_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kwargs)
            else:
                raise ValueError(f"Unknown scheduler_name: {self.scheduler_name}")

        logs = {'train_loss': [], 'val_loss': [], 'early_stopped': False}
        start_time = time.time()
        best_val_loss = float('inf')
        best_state_dict = None
        patience_counter = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_train = 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.input_noise_std > 0:
                    xb = xb + torch.randn_like(xb) * self.input_noise_std

                optimizer.zero_grad()

                if str(self.optimizer_name).lower() == "lbfgs":
                    def closure():
                        optimizer.zero_grad()
                        pred = model(xb)
                        loss = _criterion(pred, yb)
                        loss.backward()
                        if self.grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.grad_clip_norm))
                        return loss.detach()

                    loss = optimizer.step(closure)
                    # LBFGS returns loss tensor
                    loss_value = float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)
                else:
                    pred = model(xb)
                    loss = _criterion(pred, yb)
                    loss.backward()
                    if self.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.grad_clip_norm))
                    optimizer.step()
                    loss_value = float(loss.detach().cpu().item())

                train_loss += loss_value
                n_train += 1
            train_loss /= n_train

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in self.val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = model(xb)
                    loss = _criterion(pred, yb)
                    val_loss += float(loss.detach().cpu().item())
                    n_val += 1
            val_loss /= n_val

            if scheduler is not None:
                if sched_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            logs['train_loss'].append(train_loss)
            logs['val_loss'].append(val_loss)

            # Track best validation checkpoint (for reproducible evaluation).
            # Apply min_delta so tiny fluctuations don't count as improvements.
            improved = val_loss < (best_val_loss - float(self.min_delta))
            if improved:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(model.state_dict())

            # Early stopping logic
            if epoch >= self.warmup:
                if improved:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logs['early_stopped'] = True
                        break

        # Restore best model parameters seen on validation.
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        end_time = time.time()
        logs['training_time'] = end_time - start_time
        return logs

def get_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get predictions and targets from a model and data loader."""
    model = model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device).float()
            preds = model(inputs)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    y_scaler = getattr(getattr(loader, "dataset", None), "y_scaler", None)
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds)
        targets = y_scaler.inverse_transform(targets)

    return preds, targets


def build_scheduler_kwargs(params: dict) -> dict:
    """Build scheduler keyword arguments from a trial-parameter dictionary."""
    scheduler_name = str(params.get("scheduler", "none")).lower()
    if scheduler_name == "plateau":
        return {
            "factor": float(params.get("plateau_factor", 0.5)),
            "patience": int(params.get("plateau_patience", 4)),
            "min_lr": float(params.get("plateau_min_lr", 1e-6)),
        }
    if scheduler_name == "cosine":
        return {
            "eta_min": float(params.get("cosine_eta_min", 1e-6)),
        }
    return {}


def build_optimizer_kwargs(params: dict) -> dict:
    """Build optimizer keyword arguments from a trial-parameter dictionary.

    Note: weight decay is omitted for LBFGS since it is unsupported.
    """
    opt_name = str(params.get("optimizer", "Adam")).lower()
    optimizer_kwargs: dict = {}

    # Weight decay is not supported by LBFGS.
    if "weight_decay" in params and opt_name != "lbfgs":
        optimizer_kwargs["weight_decay"] = float(params["weight_decay"])

    if opt_name == "sgd":
        if "sgd_momentum" in params:
            optimizer_kwargs["momentum"] = float(params["sgd_momentum"])
        if "sgd_nesterov" in params:
            optimizer_kwargs["nesterov"] = bool(params["sgd_nesterov"])

    if opt_name == "rmsprop":
        if "rmsprop_momentum" in params:
            optimizer_kwargs["momentum"] = float(params["rmsprop_momentum"])
        if "rmsprop_alpha" in params:
            optimizer_kwargs["alpha"] = float(params["rmsprop_alpha"])

    if opt_name == "lbfgs":
        if "lbfgs_max_iter" in params:
            optimizer_kwargs["max_iter"] = int(params["lbfgs_max_iter"])
        if "lbfgs_history_size" in params:
            optimizer_kwargs["history_size"] = int(params["lbfgs_history_size"])
        if "lbfgs_line_search" in params:
            optimizer_kwargs["line_search_fn"] = params["lbfgs_line_search"]

    return optimizer_kwargs


def build_model_and_trainer_from_params(
    params: dict,
    *,
    scale_method: str,
    device: str,
    loaders_factory: Callable[[str, Union[int, str], Optional[int]], tuple[Any, Any, Any]],
    model_factory: Callable[..., nn.Module],
    trainer_factory: Callable[..., Any],
) -> tuple[Any, Any, Any, nn.Module, Any]:
    """Recreate loaders/model/trainer using all available params from a trial.

    The factories are injected to keep this helper reusable and to avoid circular
    imports with scripts that define project-specific data/model wiring.
    """
    train_batch_size: Union[int, str]
    opt_name = str(params.get("optimizer", "Adam")).lower()
    raw_bs = params.get("train_batch_size", 32)
    if str(raw_bs) == "ALL":
        train_batch_size = "ALL"
    else:
        train_batch_size = int(raw_bs)

    if opt_name == "lbfgs":
        train_batch_size = "ALL"

    # Extract generator_seed if available (for diagnostics path)
    generator_seed = params.get("_generator_seed", None)
    train_loader, val_loader, test_loader = loaders_factory(scale_method, train_batch_size, generator_seed)

    model = model_factory(
        int(params.get("hidden_size", 32)),
        str(params.get("activation", "tanh")),
        dropout_p=float(params.get("dropout_p", 0.0)),
        norm=str(params.get("norm", "none")),
    )

    trainer = trainer_factory(
        str(params.get("optimizer", "Adam")),
        float(params.get("lr", 1e-3)),
        train_loader,
        val_loader,
        device,
        loss_name=str(params.get("loss", "mse")),
        scheduler_name=str(params.get("scheduler", "none")),
        scheduler_kwargs=build_scheduler_kwargs(params),
        grad_clip_norm=params.get("grad_clip_norm", None),
        input_noise_std=float(params.get("input_noise_std", 0.0)),
        **build_optimizer_kwargs(params),
    )

    return train_loader, val_loader, test_loader, model, trainer