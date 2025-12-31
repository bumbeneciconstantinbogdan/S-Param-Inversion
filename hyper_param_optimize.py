
"""
Hyperparameter Optimization for MLP Models

This module provides a comprehensive wrapper for Optuna-based hyperparameter optimization
of PyTorch MLP models. Supports both single-objective and multi-objective optimization
with constraints, parallel execution, and detailed study analysis.

Key features:
- Multi-objective optimization: Maximize accuracy, minimize model complexity
- Constraint handling: Enforce maximum error thresholds
- Parallel execution: Process-based parallelism for CPU-bound workloads
- Study persistence: SQLite-based storage for reproducible experiments
- Visualization: Pareto fronts, parameter importances, and diagnostics

Supports various optimizers, activations, regularization techniques, and learning rate schedules.
"""

from __future__ import annotations

import optuna
import torch
import torch.nn as nn
import math

from pathlib import Path
from typing import Callable, Any, Optional, Union, Sequence, Mapping
from data_utils import seed_everything

CONSTRAINT_BIG_M_FLOAT = float((2**62) - 1)

def constraints_from_user_attrs(
    trial: optuna.trial.FrozenTrial,
    *,
    key: str = "constraints",
    expected_len: int = 1,
) -> Sequence[float]:
    """Safely extract Optuna constraints from trial.user_attrs.

    Returns a list of floats of length expected_len.
    Keeps stored constraint values as-is when they are finite.
    Replaces only the non-finite entries (NaN/inf) with a large finite value.
    If missing/malformed, returns a large finite constraint vector (treated as infeasible).
    """
    n = int(expected_len)
    v = getattr(trial, "user_attrs", {}).get(key)
    if isinstance(v, (list, tuple)) and len(v) == n:
        try:
            # JSON + DB backends may represent NaN/inf inconsistently; never emit non-finite
            # values to Optuna's constrained samplers.
            return [
                (fx if math.isfinite(fx) else CONSTRAINT_BIG_M_FLOAT)
                for fx in (float(x) for x in v)
            ]
        except Exception:
            return [CONSTRAINT_BIG_M_FLOAT] * n
    return [CONSTRAINT_BIG_M_FLOAT] * n

class HyperParamOptimizer:
    """A simple wrapper for Optuna hyperparameter optimization with PyTorch models."""

    def __init__(
        self,
        model_factory: Callable[[int, str], nn.Module],
        trainer_factory: Callable[[str, float, torch.utils.data.DataLoader, torch.utils.data.DataLoader, str], Any],
        evaluator: Callable[[Any, nn.Module, torch.utils.data.DataLoader], Any],
        loaders_factory: Callable[[Union[int, str]], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
        device='cpu',
        sampler_name: str = "tpe",
        max_err_constraint: Optional[float] = None,
    ):
        """
        Args:
            model_factory: Function that takes (hidden_size, activation) and returns nn.Module
            trainer_factory: Function that takes (optimizer_name, lr, train_loader, val_loader) and returns trainer instance.
            evaluator: Function that takes (trainer, model, test_loader) and returns a float score.
            loaders_factory: Function that takes (batch_size, generator_seed) and returns (train_loader, val_loader, test_loader).
            device: Device for training
            sampler_name: "tpe" (single objective) or "nsga3"/"nsga-ii" (multi objective).
        """
        self.model_factory = model_factory
        self.trainer_factory = trainer_factory
        self.evaluator = evaluator
        self.loaders_factory = loaders_factory
        self.device = device
        self.sampler_name = sampler_name
        self.multi_objective = sampler_name.lower() in ["nsga3", "nsga-iii", "nsga-ii", "nsga2", "nsga_ii"]
        self.max_err_constraint = float(max_err_constraint) if max_err_constraint is not None else None
        self._trial_base_seed: Optional[int] = None

    @staticmethod
    def _dedupe_trials_by_params(
        trials: Sequence[optuna.trial.FrozenTrial],
        *,
        keys: Optional[Sequence[str]] = None,
        float_sig: int = 6,
    ) -> list[optuna.trial.FrozenTrial]:
        """Return trials de-duplicated by exact hyperparameter configuration.

        Key = sorted items of ``trial.params`` after normalization.
        Keeps first occurrence in the provided order.

        Notes:
        - Optuna stores raw floats; many "duplicate" rows in tables are only equal up to
          display precision (e.g. `0.00400012` vs `0.00399998`). We therefore normalize
          floats to a fixed significant-digit string representation.
        - If ``keys`` is provided, only those hyperparameters are used for the key.
        """

        def _to_py_scalar(v: Any) -> Any:
            try:
                return v.item()  # type: ignore[attr-defined]
            except Exception:
                return v

        def _norm(v: Any) -> Any:
            v = _to_py_scalar(v)
            if v is None:
                return None
            if isinstance(v, bool):
                return bool(v)
            if isinstance(v, int):
                return int(v)
            if isinstance(v, float):
                if not math.isfinite(v):
                    return "nonfinite"
                s = max(1, int(float_sig))
                return f"{float(v):.{s}g}"
            if isinstance(v, str):
                return v
            try:
                hash(v)
                return v
            except Exception:
                return repr(v)

        seen: set[tuple] = set()
        unique: list[optuna.trial.FrozenTrial] = []
        for t in trials:
            try:
                params = dict(t.params or {})
                if keys is not None:
                    params = {k: params.get(k) for k in list(keys)}
                key = tuple(sorted((k, _norm(v)) for k, v in params.items()))
            except Exception:
                key = (int(getattr(t, "number", -1)),)
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)
        return unique

    def _constraints_func(self) -> Optional[Callable[[optuna.trial.FrozenTrial], Sequence[float]]]:
        """Return Optuna constraints_func enforcing max_err <= limit (in %).

        Mirrors Optuna's recommended pattern:
        - objective stores a list under trial.user_attrs["constraints"]
        - constraints_func returns that list

        Constraint convention: values <= 0 are feasible.

        Important: constraints must always be finite (no NaN/inf). If a trial crashes before
        constraints are computed, we keep a large finite positive fallback so Optuna treats it
        as infeasible.
        """
        if self.max_err_constraint is None:
            return None
        expected_len = 1

        def _frozen_constraints(trial: optuna.trial.FrozenTrial) -> Sequence[float]:
            return constraints_from_user_attrs(
                trial,
                key="constraints",
                expected_len=expected_len,
            )

        return _frozen_constraints

    def objective(self, trial: optuna.Trial) -> Union[float, tuple[float, float]]:
        """Optuna objective function - customize hyperparameters here."""

        # Ensure the constraint attribute exists even if the trial fails mid-way.
        # (Constraints are read later by the sampler from FrozenTrials.)
        if self.max_err_constraint is not None:
            trial.set_user_attr("constraints", [CONSTRAINT_BIG_M_FLOAT])

        hidden_size = trial.suggest_categorical('hidden_size', [8, 16, 32, 64])
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'gelu', 'leaky_relu', 'elu'])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop', 'SGD', 'LBFGS'])
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        train_batch_size = trial.suggest_categorical("train_batch_size", [8, 16, 32, 64, 128, 256, 512, "ALL"])

        # Regularization / training knobs (do not change hidden-layer sizes nor max epochs).
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
        dropout_p = trial.suggest_float("dropout_p", 0.0, 0.4)
        norm = trial.suggest_categorical("norm", ["none", "layernorm", "batchnorm"])
        loss_name = trial.suggest_categorical("loss", ["mse", "smooth_l1", "eps_rel_mse"])
        grad_clip_norm = trial.suggest_categorical("grad_clip_norm", [None, 0.5, 1.0, 2.0, 5.0])
        input_noise_std = trial.suggest_float("input_noise_std", 0.0, 0.01)

        scheduler_name = trial.suggest_categorical("scheduler", ["none", "plateau", "cosine"])
        scheduler_kwargs = {}
        if scheduler_name == "plateau":
            scheduler_kwargs = {
                "factor": trial.suggest_float("plateau_factor", 0.2, 0.8),
                "patience": trial.suggest_int("plateau_patience", 2, 8),
                "min_lr": trial.suggest_float("plateau_min_lr", 1e-6, 1e-4, log=True),
            }
        elif scheduler_name == "cosine":
            scheduler_kwargs = {
                "eta_min": trial.suggest_float("cosine_eta_min", 1e-6, 1e-4, log=True),
            }

        # Optimizer-specific extras
        optimizer_kwargs = {"weight_decay": float(weight_decay)}
        opt_lower = str(optimizer_name).lower()
        if opt_lower == "sgd":
            optimizer_kwargs.update({
                "momentum": trial.suggest_float("sgd_momentum", 0.0, 0.95),
                "nesterov": trial.suggest_categorical("sgd_nesterov", [False, True]),
            })
        if opt_lower == "rmsprop":
            optimizer_kwargs.update({
                "momentum": trial.suggest_float("rmsprop_momentum", 0.0, 0.95),
                "alpha": trial.suggest_float("rmsprop_alpha", 0.85, 0.99),
            })
        if opt_lower == "lbfgs":
            # LBFGS works best with full batch; force it.
            train_batch_size = "ALL"
            # torch.optim.LBFGS does not accept weight_decay.
            optimizer_kwargs.pop("weight_decay", None)
            optimizer_kwargs.update({
                "max_iter": trial.suggest_int("lbfgs_max_iter", 10, 30),
                "history_size": trial.suggest_int("lbfgs_history_size", 10, 50),
                "line_search_fn": trial.suggest_categorical("lbfgs_line_search", [None, "strong_wolfe"]),
            })

        # Per-trial reproducibility: seed before building loaders/model/training.
        trial_seed = None
        if self._trial_base_seed is not None:
            trial_seed = int(self._trial_base_seed) + int(trial.number)
            seed_everything(trial_seed)

        train_loader, val_loader, test_loader = self.loaders_factory(train_batch_size, trial_seed)

        model = self.model_factory(hidden_size, activation, dropout_p=dropout_p, norm=norm)
        trainer = self.trainer_factory(
            optimizer_name,
            lr,
            train_loader,
            val_loader,
            self.device,
            loss_name=loss_name,
            scheduler_name=scheduler_name,
            scheduler_kwargs=scheduler_kwargs,
            grad_clip_norm=grad_clip_norm,
            input_noise_std=input_noise_std,
            **optimizer_kwargs,
        )

        result = self.evaluator(trainer, model, test_loader)
        score: float
        max_err: Optional[float]
        if isinstance(result, tuple) and len(result) >= 2:
            score = float(result[0])
            try:
                max_err = float(result[1])
            except Exception:
                max_err = None
        else:
            score = float(result)
            max_err = None

        if max_err is not None:
            max_err_f = float(max_err)
            if not math.isfinite(max_err_f):
                max_err_f = float(CONSTRAINT_BIG_M_FLOAT)

            trial.set_user_attr("max_err", max_err_f)
            if self.max_err_constraint is not None:
                trial.set_user_attr(
                    "constraints",
                    [max_err_f - float(self.max_err_constraint)],
                )

        if self.multi_objective:
            return float(score), float(hidden_size)

        return float(score)

    def optimize(
        self,
        direction: str = 'minimize',
        n_trials: int = 50,
        n_jobs: int = 1,
        study_name: str = 'study',
        seed: Optional[int] = None,
        sqlite_path: Optional[Union[str, Path]] = None,
        storage: Optional[object] = None,
        load_if_exists: bool = True,
    ):
        """Run optimization.

        Args:
            direction: Direction for single-objective ("minimize" or "maximize").
            sqlite_path: If provided, persists the Optuna study to a SQLite DB at this path.
                Example: "artifacts/optuna_study.db".
            load_if_exists: When using sqlite_path, reuse an existing study with the same
                study_name if present.
            n_jobs: Number of parallel jobs to run. Optuna uses a thread pool for n_jobs > 1.
        """
        if not isinstance(n_jobs, int) or n_jobs < 1:
            raise ValueError(f"n_jobs must be an integer >= 1, got {n_jobs!r}")

        # Store for per-trial seeding inside objective().
        self._trial_base_seed = int(seed) if seed is not None else None

        optuna_storage: Optional[object] = storage
        if optuna_storage is None and sqlite_path is not None:
            db_path = Path(sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            optuna_storage = f"sqlite:///{db_path.as_posix()}"

        if self.multi_objective:
            directions = [direction, 'minimize']
        else:
            directions = None

        if self.multi_objective:
            sampler_name = self.sampler_name.lower()
            if sampler_name in {"nsga3", "nsga-iii", "nsga_iii"}:
                sampler = optuna.samplers.NSGAIIISampler(seed=seed, constraints_func=self._constraints_func())
            elif sampler_name in {"nsga2", "nsga-ii", "nsga_ii"}:
                sampler = optuna.samplers.NSGAIISampler(seed=seed, constraints_func=self._constraints_func())
            else:
                raise ValueError(f"Invalid sampler_name for multi-objective: {self.sampler_name}")

            study = optuna.create_study(
                study_name=study_name,
                directions=list(directions),
                sampler=sampler,
                storage=optuna_storage,
                load_if_exists=load_if_exists if optuna_storage is not None else False,
            )
        else:
            sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else optuna.samplers.TPESampler()
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
            study = optuna.create_study(
                study_name=study_name,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                storage=optuna_storage,
                load_if_exists=load_if_exists if optuna_storage is not None else False,
            )

        try:
            # Don't let a single bad hyperparameter combo kill the whole run.
            study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))
        except KeyboardInterrupt:
            # Allow graceful early stop (e.g. Ctrl+C) while keeping the study object usable.
            pass
        return study

    @staticmethod
    def plot_pareto_front_2d(
        study: optuna.Study,
        *,
        save_path: Union[str, Path],
        x_label: str = "Objective 2",
        y_label: str = "Objective 1",
        title: str | None = None,
    ) -> Optional[Path]:
        """Plot a 2-objective Pareto front.

        Uses the convention:
        - x-axis = objective index 1
        - y-axis = objective index 0
        """
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        n_objectives = len(getattr(study, "directions", []) or [])
        if n_objectives != 2:
            return None

        all_trials = [t for t in study.trials if getattr(t, "values", None) is not None and t.values is not None]
        if len(all_trials) == 0:
            return None

        pareto_trials = list(getattr(study, "best_trials", []) or [])
        pareto_trials = [t for t in pareto_trials if getattr(t, "values", None) is not None and t.values is not None]
        if len(pareto_trials) == 0:
            return None

        # Separate trials into three categories: Pareto front, feasible non-Pareto, and infeasible
        pareto_set = set(t.number for t in pareto_trials)
        
        pareto_x, pareto_y = [], []
        feasible_x, feasible_y = [], []
        infeasible_x, infeasible_y = [], []
        
        for t in all_trials:
            x_val = float(t.values[1])
            y_val = float(t.values[0])
            max_err = t.user_attrs.get('max_err')
            is_feasible = max_err is not None and max_err <= 10.0
            
            if t.number in pareto_set:
                pareto_x.append(x_val)
                pareto_y.append(y_val)
            elif is_feasible:
                feasible_x.append(x_val)
                feasible_y.append(y_val)
            else:
                infeasible_x.append(x_val)
                infeasible_y.append(y_val)

        fig, ax = plt.subplots(figsize=(10, 7))
        # Plot infeasible trials in light gray with X marker
        if infeasible_x:
            ax.scatter(infeasible_x, infeasible_y, alpha=0.3, marker='x', s=40, color='#CCCCCC', label="Infeasible trials")
        # Plot feasible non-Pareto trials in darker gray with circle marker
        if feasible_x:
            ax.scatter(feasible_x, feasible_y, alpha=0.5, marker='o', s=50, color='steelblue', label="Feasible trials", edgecolors="black", linewidth=0.5)
        # Plot Pareto front in gold with diamond marker
        if pareto_x:
            ax.scatter(pareto_x, pareto_y, color="gold", s=120, marker='D', label="Pareto front", edgecolors="black", linewidth=1.5, zorder=5)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title or f"Pareto Front: {getattr(study, 'study_name', 'study')}", fontsize=12, fontweight='bold')
        # Place legend outside plot area to avoid overlap
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    @staticmethod
    def select_candidate_trials(study: optuna.Study, *, pareto_only: bool = True) -> list[optuna.trial.FrozenTrial]:
        """Return the trials to run diagnostics on.

        Rules:
        - Multi-objective: by default returns Optuna's Pareto front (``best_trials``).
          If that is empty or pareto_only=False, falls back to all completed trials.
        - Single-objective: returns ``best_trial`` if available.
        """
        directions = list(getattr(study, "directions", []) or [])
        is_multi = getattr(study, "directions", None) is not None and len(directions) > 1

        if is_multi:
            trials = list(getattr(study, "best_trials", []) or []) if pareto_only else [t for t in study.trials if t.values is not None]
            trials = [t for t in trials if getattr(t, "values", None) is not None and t.values is not None]
            if len(trials) == 0:
                trials = [t for t in study.trials if getattr(t, "values", None) is not None and t.values is not None]

            # Dedupe by the same params shown in summaries (hidden_size, train_batch_size, activation, optimizer, lr).
            # This ensures diagnostics match the deduped summary table.
            dedupe_keys = ["hidden_size", "train_batch_size", "activation", "optimizer", "lr"]
            return HyperParamOptimizer._dedupe_trials_by_params(trials, keys=dedupe_keys, float_sig=6)

        best = getattr(study, "best_trial", None)
        return [best] if best is not None else []

    @staticmethod
    def select_pareto_compromise(study: optuna.Study) -> Optional[dict[str, Any]]:
        """Pick a single "compromise" trial from the Pareto front.

        Strategy: select the trial with the highest OK@1% (objective 0).

        Returns:
            A dict with trial number, raw objective values, normalized values, and distance.
            Returns None if the study is not 2+ objective or has no Pareto trials.
        """
        directions = list(getattr(study, "directions", []) or [])
        is_multi = getattr(study, "directions", None) is not None and len(directions) > 1
        if not is_multi:
            return None

        pareto_trials = list(getattr(study, "best_trials", []) or [])
        pareto_trials = [t for t in pareto_trials if getattr(t, "values", None) is not None and t.values is not None]
        if len(pareto_trials) == 0:
            return None

        # Select the trial with the highest objective 0 (OK@1%)
        best_trial = max(pareto_trials, key=lambda t: float(t.values[0]))

        vals = list(best_trial.values)
        return {
            "trial_number": int(best_trial.number),
            "values": [float(v) for v in vals],
            "values_norm": [float(v) for v in vals],  # Not normalized, but keep for compatibility
            "distance": 0.0,  # Not used
        }

    @staticmethod
    def compute_param_importances(
        study: optuna.Study,
        *,
        objective_labels: Optional[Sequence[str]] = None,
    ) -> tuple[Optional[list[tuple[str, float]]], Optional[dict[str, list[tuple[str, float]]]]]:
        """Compute Optuna hyperparameter importances.

        For multi-objective studies, returns a mapping ``label -> importance list`` where each list
        is ``[(param_name, importance), ...]``.
        For single-objective studies, returns the single list and ``None`` for the mapping.
        """
        from optuna.importance import get_param_importances

        directions = list(getattr(study, "directions", []) or [])
        is_multi = getattr(study, "directions", None) is not None and len(directions) > 1

        if not is_multi:
            imp = get_param_importances(study, target=None)
            return ([(str(k), float(v)) for k, v in imp.items()], None)

        n_obj = len(directions)
        if objective_labels is None:
            # Default to stable, human-readable labels for 2-objective studies.
            if n_obj == 2:
                objective_labels = ["Objective 1", "Objective 2"]
            else:
                objective_labels = [f"Objective {i+1}" for i in range(n_obj)]

        out: dict[str, list[tuple[str, float]]] = {}
        for i, label in enumerate(list(objective_labels)[:n_obj]):
            tgt = (lambda idx: (lambda t: float(t.values[idx])))(i)
            imp = get_param_importances(study, target=tgt)
            out[str(label)] = [(str(k), float(v)) for k, v in imp.items()]

        first = out.get(str(list(out.keys())[0]))
        return (first, out)

    @staticmethod
    def plot_param_importances(
        *,
        importances: Optional[Sequence[tuple[str, float]]] = None,
        importances_by_objective: Optional[Mapping[str, Sequence[tuple[str, float]]]] = None,
        save_path: Union[str, Path],
        title: str = "Hyperparameter importance (Optuna)",
        top_k: int = 15,
    ) -> Optional[Path]:
        """Plot hyperparameter importances (single or multi-objective).

        If ``importances_by_objective`` is provided, creates one subplot per objective.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return None

        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if importances_by_objective is not None and len(importances_by_objective) > 0:
            items = list(importances_by_objective.items())
            
            # Collect all unique parameters and their importances per objective
            all_params = set()
            for label, pairs in items:
                for k, v in (pairs or []):
                    all_params.add(str(k))
            
            # Build importance dict per parameter, per objective
            param_importances = {param: {} for param in all_params}
            for label, pairs in items:
                for k, v in (pairs or []):
                    param_importances[str(k)][label] = float(v)
            
            # Rank by maximum importance across all objectives
            ranked_params = sorted(
                param_importances.items(),
                key=lambda x: max(x[1].values()) if x[1] else 0,
                reverse=True
            )[:int(top_k)]
            
            param_names = [k for k, _ in ranked_params][::-1]
            n_params = len(param_names)
            
            # Color scheme for objectives
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            fig, ax = plt.subplots(figsize=(9.0, max(4.0, 0.35 * n_params)))
            
            # Plot grouped bars for each parameter
            bar_width = 0.8 / len(items)
            y_positions = np.arange(n_params)
            
            for idx, (label, pairs) in enumerate(items):
                importance_dict = dict(pairs or [])
                values = [importance_dict.get(param, 0.0) for param in param_names]
                offset = (idx - len(items)/2 + 0.5) * bar_width
                ax.barh(y_positions + offset, values, bar_width, 
                       label=label, color=colors[idx % len(colors)], alpha=0.85)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(param_names)
            ax.set_xlabel("Importance", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            
            fig.tight_layout()
            fig.savefig(out, dpi=200)
            plt.close(fig)
            return out

        ranked = sorted([(str(k), float(v)) for k, v in (importances or [])], key=lambda kv: kv[1], reverse=True)
        topk = ranked[: int(top_k)]
        names = [k for k, _ in topk][::-1]
        values = [float(v) for _, v in topk][::-1]
        fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(names))))
        ax.barh(names, values)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out

    @staticmethod
    def summarize_study_from_sqlite(
        sqlite_path: Union[str, Path],
        study_name: str,
        *,
        pareto_only: bool = True,
        top_n: int = 50,
        objective_names: Optional[Sequence[str]] = None,
        param_names: Optional[Sequence[str]] = None,
        plot: bool = False,
        plot_save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[optuna.Study]:
        """Load a study from a SQLite DB and print a generic summary.

        Works for both single-objective and multi-objective studies.

        Args:
            sqlite_path: Path to the SQLite DB used by Optuna.
            study_name: Optuna study name inside the DB.
            pareto_only: For multi-objective studies, show only the Pareto front
                (Optuna's ``best_trials``). If False, uses all completed trials.
            top_n: Maximum number of trials to print in the table.
            objective_names: Optional labels for objectives, in order.
            param_names: Optional list of parameter names to show as columns.
                If omitted, shows a stable selection of common keys.
            plot: If True, generate a plot of the study results (Pareto front for multi-obj).
            plot_save_path: Path to save the plot image. If None and plot=True, saves to
                the same directory as sqlite_path with name '{study_name}_plot.png'.

        Returns:
            The loaded ``optuna.Study`` instance.
        """
        db_path = Path(sqlite_path)
        storage = f"sqlite:///{db_path.as_posix()}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except KeyError:
            print(f"Study '{study_name}' not found in {db_path.as_posix()}.")
            return None

        n_objectives = len(getattr(study, "directions", []) or [])
        is_multi = n_objectives > 1

        if is_multi:
            if pareto_only:
                trials = list(getattr(study, "best_trials", []) or [])
            else:
                trials = [t for t in study.trials if t.values is not None]
        else:
            trials = [study.best_trial] if getattr(study, "best_trial", None) is not None else []

        if len(trials) == 0:
            print(f"No trials found for study '{study_name}'.")
            return study

        # Objective labels
        if objective_names is None:
            if is_multi:
                objective_names = [f"obj{i}" for i in range(n_objectives)]
            else:
                objective_names = ["objective"]

        # Parameter columns
        if param_names is None:
            preferred = ["hidden_size", "train_batch_size", "activation", "optimizer", "lr"]
            present = []
            for key in preferred:
                if any(key in t.params for t in trials):
                    present.append(key)
            if len(present) == 0:
                all_keys = sorted({k for t in trials for k in t.params.keys()})
                present = all_keys[:8]
            param_names = present

        # Sorting
        def _direction_is_max(i: int) -> bool:
            try:
                return str(study.directions[i]).lower().endswith("maximize")
            except Exception:
                return False

        if is_multi:
            def sort_key(t: optuna.trial.FrozenTrial):
                values = list(t.values)
                key_parts = []
                for i, v in enumerate(values):
                    if v is None:
                        key_parts.append(CONSTRAINT_BIG_M_FLOAT)
                        continue
                    key_parts.append(-float(v) if _direction_is_max(i) else float(v))
                return tuple(key_parts)

            trials_sorted = sorted(trials, key=sort_key)
        else:
            trials_sorted = trials

        # Deduplicate by the displayed param columns only (other hidden params may differ).
        trials_sorted = HyperParamOptimizer._dedupe_trials_by_params(
            trials_sorted, keys=param_names, float_sig=6
        )

        # Print summary table
        print(f"Study: {study_name}")
        print(f"Storage: {db_path.as_posix()}")
        print(f"Multi-objective: {is_multi}")

        col_headers = ["trial"] + list(objective_names) + list(param_names)
        widths = {"trial": 7}
        for name in objective_names:
            widths[name] = max(10, len(str(name)) + 2)
        for name in param_names:
            widths[name] = max(12, len(str(name)) + 2)

        header = "  ".join(f"{h:>{widths.get(h, 12)}}" for h in col_headers)
        print(header)
        print("-" * len(header))

        for t in trials_sorted[: max(1, int(top_n))]:
            row = [f"{t.number}".rjust(widths["trial"]) ]
            if is_multi:
                vals = list(t.values)
            else:
                vals = [t.value]
            for i, name in enumerate(objective_names):
                v = vals[i] if i < len(vals) else None
                if v is None:
                    cell = "".rjust(widths[name])
                else:
                    cell = f"{float(v):.6g}".rjust(widths[name])
                row.append(cell)
            for name in param_names:
                v = t.params.get(name, "")
                if isinstance(v, float):
                    cell = f"{v:.6g}".rjust(widths[name])
                else:
                    cell = str(v).rjust(widths[name])
                row.append(cell)
            print("  ".join(row))

        # Optional plotting
        if plot:
            try:
                if is_multi and n_objectives == 2:
                    save_path = plot_save_path or Path(sqlite_path).parent / f"{study_name}_pareto_front.png"
                    out = HyperParamOptimizer.plot_pareto_front_2d(
                        study,
                        save_path=save_path,
                        x_label=objective_names[1] if objective_names else "Objective 2",
                        y_label=objective_names[0] if objective_names else "Objective 1",
                        title=f"Pareto Front: {study_name}",
                    )
                    if out is not None:
                        print(f"Plot saved to: {out}")
                else:
                    print("Plotting is only supported for 2-objective multi-objective studies.")
            except ImportError:
                print("Matplotlib not available; skipping plot.")
            except Exception as e:
                print(f"Plotting failed: {e}")

        return study