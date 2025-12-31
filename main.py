"""
MLP Hyperparameter Optimization for Permittivity Prediction

This script performs multi-objective hyperparameter optimization using Optuna
to train MLP models for predicting complex permittivity from S-parameters.

Key features:
- Multi-objective optimization: Maximize OK@1% accuracy, minimize hidden layer width
- Constraint: Max error ≤ 10%
- Parallel execution support
- Comprehensive diagnostics and visualization
- NRW baseline comparison

Usage:
    python main.py  # Run with SUMMARY_ONLY=True to analyze existing study
    # Edit SUMMARY_ONLY=False to run new HPO
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional
import os
import multiprocessing as mp
from pathlib import Path
import optuna
from optuna.storages import RDBStorage
from mlp import MLPRegressor, Trainer, get_predictions, build_model_and_trainer_from_params
from hyper_param_optimize import HyperParamOptimizer, constraints_from_user_attrs
from data_utils import (
    create_dataloaders,
    analyze_permittivity_error,
    compute_relative_error,
    compute_ok_classification,
    convert_n2_tensor_to_complex,
    r2_score,
    plot_loss_history,
    plot_error_histograms,
    plot_r2_scatter,
    run_nrw_diagnostics,
    seed_everything,
    generate_non_magnetic_data,
    save_to_csv,
)

DATA_FILES = (
    os.path.join(os.path.dirname(__file__), "data", "data_train.csv"),
    os.path.join(os.path.dirname(__file__), "data", "data_val.csv"),
    os.path.join(os.path.dirname(__file__), "data", "data_test.csv"),
)
seed = 42
MAX_ERR_CONSTRAINT_PERCENT = 10.0


def run_top_trials_diagnostics(
    study: optuna.Study,
    *,
    scale_method: str,
    device: str,
    max_epochs: int = 150,
    thresholds: tuple[int, ...] = (10, 1),
    pareto_only: bool = True,
    save_figures: bool = True,
) -> None:
    """Run comprehensive diagnostics on top-performing trials from an Optuna study.

    This function evaluates the best trials (Pareto front for multi-objective studies,
    or the single best trial for single-objective studies) by retraining each model
    on the full training set and generating detailed analysis artifacts.

    Generated artifacts include:
    - Loss history plots (training/validation loss vs. epoch).
    - Error surface and classification maps at specified thresholds.
    - R² scatter plots for real and imaginary permittivity components.
    - Histograms of relative and absolute error distributions.

    Args:
        study: The Optuna study containing completed trials.
        scale_method: Data scaling method ("standard", "minmax", or "none").
        device: Device for model training and inference ("cpu", "cuda", etc.).
        max_epochs: Maximum epochs to train each model (default: 150).
        thresholds: Error thresholds (%) for analysis (default: (10, 1)).
        pareto_only: For multi-objective studies, evaluate only Pareto front trials.
                     For single-objective, ignored (always evaluates best trial).
        save_figures: Whether to save diagnostic plots to artifacts/ directory.

    Note:
        - For multi-objective studies, defaults to Pareto front unless pareto_only=False.
        - Models are retrained from scratch using trial parameters.
        - Artifacts are saved with trial-specific naming for easy identification.
    """
    candidate_trials = HyperParamOptimizer.select_candidate_trials(study, pareto_only=bool(pareto_only))

    if len(candidate_trials) == 0:
        print("[diagnostics] No trials found to analyze.")
        return

    artifacts_dir = Path(os.path.dirname(__file__)) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Pareto-front visualization (multi-objective only).
    directions = list(getattr(study, "directions", []) or [])
    is_multi = getattr(study, "directions", None) is not None and len(directions) > 1
    if is_multi:
        try:
            safe_study = str(getattr(study, "study_name", "study")).replace(os.sep, "_")
            fig_path = artifacts_dir / f"{safe_study}_pareto_front.png"
            HyperParamOptimizer.plot_pareto_front_2d(
                study,
                save_path=fig_path,
                x_label="Hidden width H (minimize)",
                y_label="OK@1% (maximize)",
                title="Pareto front (Optuna study)",
            )
        except Exception as e:
            print(f"[diagnostics] Could not plot Pareto front: {e}")

    for trial in candidate_trials:
        # Per-trial reproducibility: align RNG state with HPO's per-trial seeding.
        trial_seed = seed + trial.number
        seed_everything(trial_seed)
        best_params = trial.params
        train_loader, val_loader, test_loader, model, trainer = build_model_and_trainer_from_params(
            best_params,
            scale_method=scale_method,
            device=device,
            loaders_factory=lambda s, bs, gs=None: loaders_factory(s, bs, trial_seed),
            model_factory=model_factory,
            trainer_factory=trainer_factory,
        )

        logs = trainer.train(model, max_epochs=int(max_epochs))
        preds, targets = get_predictions(model, test_loader, device)

        base_name = f"pareto_trial_{trial.number}_4{best_params.get('hidden_size', 'NA')}2"

        # 1) Loss history (loss vs epoch)
        if save_figures:
            plot_loss_history(
                logs,
                artifacts_dir / f"{base_name}_loss_history.png",
                title=f"{base_name} — Loss history",
            )

        # 2) Error analysis figures
        for i, thr in enumerate(thresholds):
            analyze_permittivity_error(
                targets,
                preds,
                threshold=float(thr),
                save_figures=bool(save_figures),
                model_name=f"{base_name}_{int(thr)}",
                plot_error_surface=bool(i == 0),
                verbose=True,
            )

        # 3) R2 for eps' and eps'' on TEST
        targets_np = targets.detach().cpu().numpy() if torch.is_tensor(targets) else np.asarray(targets)
        preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)
        r2_real = r2_score(targets_np[:, 0], preds_np[:, 0])
        r2_imag = r2_score(targets_np[:, 1], preds_np[:, 1])

        if save_figures:
            plot_r2_scatter(
                y_true=targets_np,
                y_pred=preds_np,
                r2_real=r2_real,
                r2_imag=r2_imag,
                save_path=artifacts_dir / f"{base_name}_r2_scatter.png",
                title=f"{base_name} — Test pred vs true",
            )

        # 4) Relative + absolute error distributions (on complex epsilon)
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

    # Hyperparameter importance (study-level) — computed once per diagnostics run.
    try:
        objective_labels = None
        if is_multi and len(directions) == 2:
            objective_labels = ["Objective 1 (OK@1% accuracy)", "Objective 2 (hidden width H)"]

        importances, importances_by_objective = HyperParamOptimizer.compute_param_importances(
            study,
            objective_labels=objective_labels,
        )

        if save_figures and (
            (importances_by_objective and len(importances_by_objective) > 0)
            or (importances and len(importances) > 0)
        ):
            safe_study = str(getattr(study, "study_name", "study")).replace(os.sep, "_")
            fig_path = artifacts_dir / (
                f"{safe_study}_hyperparam_importance_multi.png" if importances_by_objective else f"{safe_study}_hyperparam_importance.png"
            )
            HyperParamOptimizer.plot_param_importances(
                importances=importances,
                importances_by_objective=importances_by_objective,
                save_path=fig_path,
                title="Hyperparameter importance (Optuna)",
            )
    except Exception as e:
        print(f"[diagnostics] Could not compute hyperparameter importance: {e}")

    print("[diagnostics] Completed trial diagnostics. Figures saved to artifacts/")


# Set seed for reproducibility
seed_everything(seed)


def loaders_factory(scale_method: str, train_batch_size: Union[int, str], generator_seed: Optional[int] = None) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        scale_method (str): Scaling method for the data ("standard", "minmax", or "none").
        train_batch_size (Union[int, str]): Batch size for training loader, or "ALL" for full batch.
        generator_seed (Optional[int]): Seed for reproducible data shuffling.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader, *_ = create_dataloaders(
        files=DATA_FILES,
        batch_sizes=(train_batch_size, "ALL", "ALL"),
        scale_method=scale_method,
        generator_seed=generator_seed,
    )
    return train_loader, val_loader, test_loader

def model_factory(hidden_size: int, activation: str, *, dropout_p: float = 0.0, norm: str = "none") -> nn.Module:
    """
    Create an MLPRegressor model.

    Args:
        hidden_size (int): Number of neurons in the hidden layer.
        activation (str): Activation function name.

    Returns:
        nn.Module: The MLP model.
    """
    return MLPRegressor(
        input_size=4,
        output_size=2,
        hidden_size=hidden_size,
        activation=activation,
        dropout_p=dropout_p,
        norm=norm,
    )

def trainer_factory(
    optimizer_name: str,
    lr: float,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    **kwargs,
):
    """
    Create a Trainer instance.

    Args:
        optimizer_name (str): Name of the optimizer.
        lr (float): Learning rate.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device (str): Device for training.

    Returns:
        Trainer: The trainer instance.
    """
    return Trainer(train_loader, val_loader, optimizer_name, lr, device=device, patience=5, warmup=5, **kwargs)

def evaluator(trainer: Trainer, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> tuple[float, float]:
    """
    Evaluate the model on the test set and return the OK percentage at the 1% threshold.

    Args:
        trainer: Trainer instance to train the model.
        model: The model to evaluate.
        test_loader: Test data loader.

    Returns:
        Tuple (ok_percent_1, max_err_percent) on the evaluation split.
    """
    trainer.train(model, max_epochs=25)
    preds, targets = get_predictions(model, test_loader, trainer.device)
    eps_true_complex = convert_n2_tensor_to_complex(targets)
    eps_pred_complex = convert_n2_tensor_to_complex(preds)
    rel_error, n_samples, _, max_error, _ = compute_relative_error(eps_true_complex, eps_pred_complex)
    _, _, ok_percent_1 = compute_ok_classification(rel_error, threshold=1, n_samples=n_samples)
    return float(ok_percent_1), float(max_error)


def run_hpo_worker(
    worker_id: int,
    n_trials: int,
    *,
    scale_method: str,
    sqlite_db_path: str,
    study_name: str,
    seed: int,
    device: str,
    sampler_name: str,
) -> None:
    """Run a slice of Optuna trials in a separate OS process.

    Each worker loads the same Optuna study through shared storage and evaluates trials
    sequentially (`n_jobs=1`). This gives true parallelism for CPU-bound work.
    """
    worker_seed = int(seed)
    seed_everything(worker_seed)

    loaders_factory_for_opt = lambda train_batch_size, generator_seed=None: loaders_factory(scale_method, train_batch_size, generator_seed)

    optimizer = HyperParamOptimizer(
        model_factory=model_factory,
        trainer_factory=trainer_factory,
        evaluator=evaluator,
        loaders_factory=loaders_factory_for_opt,
        device=device,
        sampler_name=sampler_name,
        max_err_constraint=MAX_ERR_CONSTRAINT_PERCENT,
    )

    # Workers must NOT race on creating Optuna's internal tables.
    # Parent process initializes schema/study once; workers attach with skip_table_creation=True.
    Path(sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{Path(sqlite_db_path).as_posix()}"
    worker_storage = RDBStorage(
        url=url,
        engine_kwargs={"connect_args": {"timeout": 60}},
        heartbeat_interval=30,
        grace_period=120,
        skip_table_creation=True,
    )

    optimizer.optimize(
        n_trials=int(n_trials),
        n_jobs=1,
        direction='maximize',
        study_name=study_name,
        seed=worker_seed,
        sqlite_path=None,
        storage=worker_storage,
        load_if_exists=True,
    )

if __name__ == "__main__":
    try:
        # Configuration via environment variables with defaults
        device = os.environ.get('DEVICE', 'cpu')  # Device to run on ('cpu' or 'cuda' or 'mps').
        SCALE_METHOD = os.environ.get('SCALE_METHOD', 'standard')  # Scaling options: "standard", "minmax", or "none"
        SUMMARY_ONLY = os.environ.get('SUMMARY_ONLY', 'True').lower() == 'true'  # If True, just summarize an existing study and exit.
        REGENERATE_DATA = os.environ.get('REGENERATE_DATA', 'True').lower() == 'true'  # If True, regenerate synthetic data before running

        # Parallel execution settings
        # - process: spawn multiple processes that share the same Optuna study storage
        # - thread: Optuna's internal thread pool via n_jobs
        OPTUNA_PARALLEL_MODE = os.environ.get('OPTUNA_PARALLEL_MODE', 'process')  # "process" or "thread" or "none"
        OPTUNA_N_PROCESSES = int(os.environ.get('OPTUNA_N_PROCESSES', '8'))
        OPTUNA_N_JOBS = int(os.environ.get('OPTUNA_N_JOBS', '1'))
        OPTUNA_N_TRIALS = int(os.environ.get('OPTUNA_N_TRIALS', '10000'))

        loaders_factory_for_opt = lambda train_batch_size, generator_seed=None: loaders_factory(SCALE_METHOD, train_batch_size, generator_seed)

        sqlite_db_path = os.path.join(os.path.dirname(__file__), "artifacts", "study.db")
        Path(sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        study_name = 'mlp_hpo_ok1_mo'
        sampler_name = "nsga2"

        # Optional: regenerate synthetic data
        if REGENERATE_DATA:
            print("Regenerating synthetic data...")
            data = generate_non_magnetic_data()
            save_to_csv(data[0], DATA_FILES[0])
            save_to_csv(data[1], DATA_FILES[1])
            save_to_csv(data[2], DATA_FILES[2])
            print("Data regeneration complete.")

        # Optional: just inspect the persisted study and exit (no training).
        if SUMMARY_ONLY:
            run_nrw_diagnostics(
                data_file=DATA_FILES[2],
                d=1.5e-3,
                a=22.86e-3,
                frequency=8.2e9,
                thresholds=(10.0, 1.0),
                save_figures=True,
                artifacts_dir=os.path.join(os.path.dirname(__file__), "artifacts"),
            )


            study = HyperParamOptimizer.summarize_study_from_sqlite(
                sqlite_path=sqlite_db_path,
                study_name=study_name,
                pareto_only=True,
                top_n=50,
                plot=True,
            )
            if study is None:
                raise SystemExit(1)
            

            run_top_trials_diagnostics(
                study,
                scale_method=SCALE_METHOD,
                device=device,
                max_epochs=25,
                thresholds=(10, 1),
                pareto_only=True,
                save_figures=True,
            )
            raise SystemExit(0)

        optimizer = HyperParamOptimizer(
            model_factory=model_factory,
            trainer_factory=trainer_factory,
            evaluator=evaluator,
            loaders_factory=loaders_factory_for_opt,
            device=device,
            sampler_name=sampler_name,
            max_err_constraint=MAX_ERR_CONSTRAINT_PERCENT,
        )

        # Build an Optuna storage object once in the parent process.
        # This initializes the schema safely and also enables heartbeats so stale RUNNING trials
        # can be handled correctly across interrupts.
        storage_url = f"sqlite:///{Path(sqlite_db_path).as_posix()}"
        parent_storage = RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 60}},
            heartbeat_interval=30,
            grace_period=120,
            skip_table_creation=False,
        )

        # IMPORTANT: initialize (or load) the study ONCE before spawning workers.
        # This avoids the SQLite DDL race that produced "table already exists" after Ctrl+C.
        if optimizer.multi_objective:
            sampler = optuna.samplers.NSGAIISampler(
                seed=seed,
                constraints_func=(
                    lambda t: constraints_from_user_attrs(
                        t,
                        key="constraints",
                        expected_len=1,
                    )
                ),
            )
            optuna.create_study(
                study_name=study_name,
                directions=['maximize', 'minimize'],
                sampler=sampler,
                storage=parent_storage,
                load_if_exists=True,
            )
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
            optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=sampler,
                pruner=pruner,
                storage=parent_storage,
                load_if_exists=True,
            )

        if OPTUNA_PARALLEL_MODE == "process" and OPTUNA_N_PROCESSES > 1:
            n_procs = int(OPTUNA_N_PROCESSES)
            total = int(OPTUNA_N_TRIALS)
            base = total // n_procs
            rem = total % n_procs
            trials_per_worker = [base + (1 if i < rem else 0) for i in range(n_procs)]

            ctx = mp.get_context("spawn")
            procs: list[mp.Process] = []
            for worker_id, n_worker_trials in enumerate(trials_per_worker):
                if n_worker_trials <= 0:
                    continue
                p = ctx.Process(
                    target=run_hpo_worker,
                    args=(worker_id, n_worker_trials),
                    kwargs={
                        "scale_method": SCALE_METHOD,
                        "sqlite_db_path": sqlite_db_path,
                        "study_name": study_name,
                        "seed": seed,
                        "device": device,
                        "sampler_name": sampler_name,
                    },
                )
                p.start()
                procs.append(p)

            interrupted = False
            try:
                # Join with timeouts so Ctrl+C can be handled promptly.
                while any(p.is_alive() for p in procs):
                    for p in procs:
                        p.join(timeout=0.2)
            except KeyboardInterrupt:
                interrupted = True
                print("\n[main] Ctrl+C received: terminating Optuna workers...")
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                for p in procs:
                    p.join(timeout=5)

            for p in procs:
                if p.exitcode not in (0, None):
                    # If interrupted, ignore non-zero worker exit codes.
                    if not interrupted:
                        raise SystemExit(p.exitcode)

            study = optuna.load_study(study_name=study_name, storage=parent_storage)

            if interrupted:
                print(f"[main] Interrupted. Study is saved at {sqlite_db_path} (name='{study_name}').")
                print("[main] Re-run main.py to continue optimization.")
                raise SystemExit(130)
        else:
            study = optimizer.optimize(
                n_trials=OPTUNA_N_TRIALS,
                n_jobs=OPTUNA_N_JOBS,
                direction='maximize',
                study_name=study_name,
                seed=seed,
                sqlite_path=sqlite_db_path,
            )

        # After HPO finishes, run the same diagnostics used in SUMMARY_ONLY.
        run_top_trials_diagnostics(
            study,
            scale_method=SCALE_METHOD,
            device=device,
            max_epochs=25,
            thresholds=(10, 1),
            pareto_only=bool(optimizer.multi_objective),
            save_figures=True,
        )

    except KeyboardInterrupt:
        # Fallback: if something outside the worker join loop is interrupted.
        print("\n[main] Interrupted. Exiting cleanly.")
        raise SystemExit(130)