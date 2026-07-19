//! Terminal-friendly HPO summary formatting.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use optimizer::TrialState;
use optimizer::parameter::ParamValue;
use optimizer::sampler::CompletedTrial as OptimizerCompletedTrial;

use crate::search_space::{
    ActivationChoice, BatchSizeChoice, HyperParams, LossChoice,
    NormChoice, OptimizerChoice, SchedulerChoice,
};
use crate::pareto::{CompletedTrial, MultiObjectiveResults};
use crate::evaluation::TrialMetrics;
use crate::{Direction, Study, TrialStatus};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configures how summaries are rendered.
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    /// Maximum number of rows shown in the table.
    pub top_n: usize,
    /// Use Unicode markers and headers when true.
    pub use_unicode: bool,
    /// Show extra trial metrics columns when available.
    pub show_metrics: bool,
    /// Columns to suppress from the otherwise full-visibility table.
    pub force_hide_params: Vec<SummaryParamColumn>,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            top_n: 20,
            use_unicode: true,
            show_metrics: false,
            force_hide_params: Vec::new(),
        }
    }
}

/// Parameter/metric columns that may be hidden manually.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SummaryParamColumn {
    HiddenSize,
    LearningRate,
    Activation,
    Optimizer,
    Scheduler,
    BatchSize,
    Norm,
    Dropout,
    WeightDecay,
    Loss,
    GradClip,
    InputNoise,
    PlateauFactor,
    PlateauPatience,
    PlateauMinLr,
    CosineEtaMin,
    SgdMomentum,
    SgdNesterov,
    RmspropMomentum,
    RmspropAlpha,
    TrainingTime,
    BestEpoch,
    FinalEpoch,
    MaxError,
}

/// Metadata displayed in the summary header/footer.
#[derive(Debug, Clone, Copy)]
pub struct SummaryMeta<'a> {
    pub study_name: &'a str,
    pub total_duration: Option<Duration>,
}

impl<'a> SummaryMeta<'a> {
    /// Create metadata with just the study name.
    #[must_use]
    pub fn new(study_name: &'a str) -> Self {
        Self {
            study_name,
            total_duration: None,
        }
    }

    /// Attach overall study duration for the footer line.
    #[must_use]
    pub fn with_total_duration(mut self, total_duration: Duration) -> Self {
        self.total_duration = Some(total_duration);
        self
    }
}

/// Unified input for single- and multi-objective summaries.
pub enum SummaryInput<'a> {
    SingleObjective {
        meta: SummaryMeta<'a>,
        study: &'a Study<f64>,
    },
    MultiObjective {
        meta: SummaryMeta<'a>,
        results: &'a MultiObjectiveResults,
    },
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Format an HPO summary as a string.
#[must_use]
pub fn format_summary(input: SummaryInput<'_>, config: &SummaryConfig) -> String {
    match input {
        SummaryInput::SingleObjective { meta, study } => format_single_objective_summary(
            meta,
            study.direction(),
            &study.trials(),
            study.n_pruned_trials(),
            config,
        ),
        SummaryInput::MultiObjective { meta, results } => {
            format_multi_objective_summary(meta, results, config)
        }
    }
}

use sparam_training::logger::{LogMessage, LogSender};

/// Print an HPO summary through the given [`LogSender`].
pub fn print_summary(log: &LogSender, input: SummaryInput<'_>, config: &SummaryConfig) {
    log.send(LogMessage::Info(format_summary(input, config)));
}

// ---------------------------------------------------------------------------
// Private symbols helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Symbols<'a> {
    pareto: &'a str,
    ok: &'a str,
    infeasible: &'a str,
    error: &'a str,
    pruned: &'a str,
    less_equal: &'a str,
}

impl Symbols<'_> {
    fn new(use_unicode: bool) -> Self {
        if use_unicode {
            Self {
                pareto: "★",
                ok: "✓ ok",
                infeasible: "✗ lim",
                error: "✗ err",
                pruned: "✗ prn",
                less_equal: "≤",
            }
        } else {
            Self {
                pareto: "*",
                ok: "[ok]",
                infeasible: "[lim]",
                error: "[err]",
                pruned: "[prn]",
                less_equal: "<=",
            }
        }
    }
}

// --- format helpers ---

fn compare_value(left: f64, right: f64, direction: Direction) -> Ordering {
    match direction {
        Direction::Maximize => right.partial_cmp(&left).unwrap_or(Ordering::Equal),
        Direction::Minimize => left.partial_cmp(&right).unwrap_or(Ordering::Equal),
    }
}

fn trial_bucket(is_completed: bool, is_feasible: bool) -> u8 {
    match (is_completed, is_feasible) {
        (true, true) => 0,
        (true, false) => 1,
        (false, _) => 2,
    }
}

fn is_constraint_feasible(constraints: &[f64]) -> bool {
    constraints.iter().all(|value| *value <= 0.0)
}

fn format_total_duration(total_duration: Option<Duration>) -> String {
    total_duration
        .map(|duration| format!(" | Duration: {}", format_duration(duration)))
        .unwrap_or_default()
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        let minutes = (secs / 60.0).floor();
        let rem = secs - minutes * 60.0;
        format!("{}m {:.1}s", minutes as u64, rem)
    } else {
        let hours = (secs / 3600.0).floor();
        let rem = secs - hours * 3600.0;
        let minutes = (rem / 60.0).floor();
        format!("{}h {}m", hours as u64, minutes as u64)
    }
}

fn format_direction(direction: Direction) -> &'static str {
    match direction {
        Direction::Maximize => "Maximize",
        Direction::Minimize => "Minimize",
    }
}

fn format_scientific(value: f64) -> String {
    if value.is_nan() {
        return String::from("nan");
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            String::from("inf")
        } else {
            String::from("-inf")
        };
    }
    format!("{value:.1e}")
}

fn format_fixed(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return String::from("nan");
    }
    if value.is_infinite() {
        return if value.is_sign_positive() {
            String::from("inf")
        } else {
            String::from("-inf")
        };
    }
    format!("{value:.precision$}")
}

fn format_percent(value: f64) -> String {
    format_fixed(value, 2)
}

fn format_noise(value: f64) -> String {
    if value >= 0.01 {
        format_fixed(value, 2)
    } else {
        format_fixed(value, 3)
    }
}

/// Render the complexity objective (trainable parameter count) as an
/// integer when representable, falling back to a two-decimal display
/// for the placeholder ±∞ / NaN sentinel values used on failed trials.
fn format_param_count(value: f64) -> String {
    if value.is_finite() && (value.fract().abs() < 1e-9) {
        format!("{}", value as i64)
    } else {
        format_fixed(value, 2)
    }
}

fn format_generic_value(value: f64) -> String {
    if value.is_finite() && value.abs() >= 1e-3 && value.abs() < 1e4 {
        format_fixed(value, 4)
    } else {
        format_scientific(value)
    }
}

fn format_indexed_choice<T>(index: usize, from_index: impl FnOnce(usize) -> Option<T>) -> String
where
    T: std::fmt::Display,
{
    from_index(index)
        .map(|choice| choice.to_string())
        .unwrap_or_else(|| String::from("?"))
}

fn format_clip_index(index: usize) -> String {
    match index {
        0 => String::from("-"),
        1 => String::from("0.5"),
        2 => String::from("1.0"),
        3 => String::from("2.0"),
        4 => String::from("5.0"),
        _ => String::from("?"),
    }
}

fn format_constraint(threshold: Option<f64>, less_equal: &str) -> String {
    threshold
        .map(|value| format!("max_err {less_equal} {}%", format_fixed(value, 1)))
        .unwrap_or_else(|| String::from("none"))
}

// --- render ---

type Cell = Cow<'static, str>;

#[derive(Debug, Clone, Copy)]
enum Align {
    Left,
    Right,
}

struct Column {
    header: Cell,
    align: Align,
    cells: Vec<Cell>,
}

fn render_summary(header: String, directions: String, columns: Vec<Column>, footer: String) -> String {
    let table = render_table(columns);
    let table_width = table.lines().map(display_width).max().unwrap_or(0);
    let width = [
        display_width(&header),
        display_width(&directions),
        display_width(&footer),
        table_width,
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .max(80);
    let border = "=".repeat(width);

    format!("{border}\n{header}\n{directions}\n{border}\n{table}\n{footer}\n{border}")
}

fn render_table(columns: Vec<Column>) -> String {
    if columns.is_empty() {
        return String::from("No trials available.");
    }

    let widths: Vec<_> = columns
        .iter()
        .map(|column| {
            std::iter::once(column.header.as_ref())
                .chain(columns_cells_as_str(&column.cells))
                .map(display_width)
                .max()
                .unwrap_or(0)
        })
        .collect();

    let total_width = widths.iter().sum::<usize>() + widths.len().saturating_sub(1);
    let row_count = columns.first().map_or(0, |column| column.cells.len());
    let line_count = if row_count == 0 { 4 } else { row_count + 3 };
    let mut out = String::with_capacity(line_count * (total_width + 1));

    write_table_row(&mut out, &columns, &widths, None);
    out.push('\n');
    push_repeated(&mut out, '-', total_width);
    out.push('\n');

    if row_count == 0 {
        out.push_str("No trials available.");
        out.push('\n');
    } else {
        for row_idx in 0..row_count {
            write_table_row(&mut out, &columns, &widths, Some(row_idx));
            if row_idx + 1 < row_count {
                out.push('\n');
            }
        }
        out.push('\n');
    }

    push_repeated(&mut out, '-', total_width);
    out
}

fn write_table_row(out: &mut String, columns: &[Column], widths: &[usize], row_idx: Option<usize>) {
    for (idx, (column, width)) in columns.iter().zip(widths.iter()).enumerate() {
        if idx > 0 {
            out.push(' ');
        }
        let cell = match row_idx {
            Some(row_idx) => column.cells[row_idx].as_ref(),
            None => column.header.as_ref(),
        };
        write_aligned(out, cell, *width, column.align);
    }
}

fn write_aligned(out: &mut String, value: &str, width: usize, align: Align) {
    let pad_len = width.saturating_sub(display_width(value));
    match align {
        Align::Left => {
            out.push_str(value);
            push_spaces(out, pad_len);
        }
        Align::Right => {
            push_spaces(out, pad_len);
            out.push_str(value);
        }
    }
}

fn display_width(value: &str) -> usize {
    if value.is_ascii() {
        value.len()
    } else {
        value.chars().count()
    }
}

fn columns_cells_as_str(cells: &[Cell]) -> impl Iterator<Item = &str> {
    cells.iter().map(Cow::as_ref)
}

fn borrowed_dash() -> Cell {
    "-".into()
}

fn push_spaces(out: &mut String, count: usize) {
    push_repeated(out, ' ', count);
}

fn push_repeated(out: &mut String, ch: char, count: usize) {
    for _ in 0..count {
        out.push(ch);
    }
}

// --- row assembly ---

#[derive(Debug, Clone)]
struct SummaryRow {
    trial_id: Cell,
    pareto: Cell,
    objective0: Cell,
    objective1: Option<Cell>,
    status: Cell,
    params: ParamCells,
    metrics: Option<MetricCells>,
}

#[derive(Debug, Clone)]
struct ParamCells {
    hidden_size: Cell,
    learning_rate: Cell,
    activation: Cell,
    optimizer: Cell,
    scheduler: Cell,
    batch_size: Cell,
    norm: Cell,
    dropout: Cell,
    weight_decay: Cell,
    loss: Cell,
    grad_clip: Cell,
    input_noise: Cell,
    plateau_factor: Cell,
    plateau_patience: Cell,
    plateau_min_lr: Cell,
    cosine_eta_min: Cell,
    sgd_momentum: Cell,
    sgd_nesterov: Cell,
    rmsprop_momentum: Cell,
    rmsprop_alpha: Cell,
}

#[derive(Debug, Clone)]
struct MetricCells {
    training_time: Cell,
    best_epoch: Cell,
    final_epoch: Cell,
    max_error: Cell,
}

struct SingleTrialLookup<'a> {
    by_label: HashMap<&'a str, &'a ParamValue>,
}

impl<'a> SingleTrialLookup<'a> {
    fn new(trial: &'a OptimizerCompletedTrial<f64>) -> Self {
        let mut by_label = HashMap::with_capacity(trial.param_labels.len());
        for (param_id, label) in &trial.param_labels {
            if let Some(value) = trial.params.get(param_id) {
                by_label.insert(label.as_str(), value);
            }
        }
        Self { by_label }
    }

    fn float(&self, label: &str) -> Option<f64> {
        match self.by_label.get(label) {
            Some(ParamValue::Float(value)) => Some(*value),
            _ => None,
        }
    }

    fn int(&self, label: &str) -> Option<i64> {
        match self.by_label.get(label) {
            Some(ParamValue::Int(value)) => Some(*value),
            _ => None,
        }
    }

    fn categorical(&self, label: &str) -> Option<usize> {
        match self.by_label.get(label) {
            Some(ParamValue::Categorical(value)) => Some(*value),
            _ => None,
        }
    }
}

fn build_multi_columns(rows: &[SummaryRow], config: &SummaryConfig) -> Vec<Column> {
    let mut columns = base_columns(rows, true);
    append_param_columns(&mut columns, rows, config, false);
    append_metric_columns(&mut columns, rows, config);
    columns
}

fn build_single_columns(rows: &[SummaryRow], config: &SummaryConfig) -> Vec<Column> {
    let mut columns = base_columns(rows, false);
    append_param_columns(&mut columns, rows, config, true);
    append_metric_columns(&mut columns, rows, config);
    columns
}

fn base_columns(rows: &[SummaryRow], is_multi: bool) -> Vec<Column> {
    let mut columns = vec![
        Column {
            header: "#".into(),
            align: Align::Right,
            cells: rows.iter().map(|row| row.trial_id.clone()).collect(),
        },
        Column {
            header: "P".into(),
            align: Align::Left,
            cells: rows.iter().map(|row| row.pareto.clone()).collect(),
        },
        Column {
            header: if is_multi { "OK@1%".into() } else { "value".into() },
            align: Align::Right,
            cells: rows.iter().map(|row| row.objective0.clone()).collect(),
        },
    ];

    if is_multi {
        columns.push(Column {
            header: "hid".into(),
            align: Align::Right,
            cells: rows
                .iter()
                .map(|row| row.objective1.clone().unwrap_or_default())
                .collect(),
        });
    }

    columns.push(Column {
        header: "status".into(),
        align: Align::Left,
        cells: rows.iter().map(|row| row.status.clone()).collect(),
    });

    columns
}

fn append_param_columns(
    columns: &mut Vec<Column>,
    rows: &[SummaryRow],
    config: &SummaryConfig,
    include_hidden_size: bool,
) {
    let mut maybe_push =
        |key: SummaryParamColumn, header: &'static str, align: Align, values: Vec<Cell>| {
            if !is_hidden(config, key) {
                columns.push(Column {
                    header: header.into(),
                    align,
                    cells: values,
                });
            }
        };

    if include_hidden_size {
        maybe_push(
            SummaryParamColumn::HiddenSize,
            "hid",
            Align::Right,
            rows.iter().map(|row| row.params.hidden_size.clone()).collect(),
        );
    }
    maybe_push(
        SummaryParamColumn::LearningRate,
        "lr",
        Align::Right,
        rows.iter()
            .map(|row| row.params.learning_rate.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::Activation,
        "act",
        Align::Left,
        rows.iter().map(|row| row.params.activation.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::Optimizer,
        "opt",
        Align::Left,
        rows.iter().map(|row| row.params.optimizer.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::Scheduler,
        "sched",
        Align::Left,
        rows.iter().map(|row| row.params.scheduler.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::BatchSize,
        "batch",
        Align::Right,
        rows.iter().map(|row| row.params.batch_size.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::Norm,
        "norm",
        Align::Left,
        rows.iter().map(|row| row.params.norm.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::Dropout,
        "drop",
        Align::Right,
        rows.iter().map(|row| row.params.dropout.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::WeightDecay,
        "wd",
        Align::Right,
        rows.iter()
            .map(|row| row.params.weight_decay.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::Loss,
        "loss",
        Align::Left,
        rows.iter().map(|row| row.params.loss.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::GradClip,
        "clip",
        Align::Right,
        rows.iter().map(|row| row.params.grad_clip.clone()).collect(),
    );
    maybe_push(
        SummaryParamColumn::InputNoise,
        "noise",
        Align::Right,
        rows.iter()
            .map(|row| row.params.input_noise.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::PlateauFactor,
        "plat_f",
        Align::Right,
        rows.iter()
            .map(|row| row.params.plateau_factor.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::PlateauPatience,
        "plat_p",
        Align::Right,
        rows.iter()
            .map(|row| row.params.plateau_patience.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::PlateauMinLr,
        "plat_lr",
        Align::Right,
        rows.iter()
            .map(|row| row.params.plateau_min_lr.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::CosineEtaMin,
        if config.use_unicode { "cos_η" } else { "cos_eta" },
        Align::Right,
        rows.iter()
            .map(|row| row.params.cosine_eta_min.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::SgdMomentum,
        "sgd_m",
        Align::Right,
        rows.iter()
            .map(|row| row.params.sgd_momentum.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::SgdNesterov,
        "nest",
        Align::Left,
        rows.iter()
            .map(|row| row.params.sgd_nesterov.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::RmspropMomentum,
        "rms_m",
        Align::Right,
        rows.iter()
            .map(|row| row.params.rmsprop_momentum.clone())
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::RmspropAlpha,
        if config.use_unicode { "rms_α" } else { "rms_alpha" },
        Align::Right,
        rows.iter()
            .map(|row| row.params.rmsprop_alpha.clone())
            .collect(),
    );
}

fn append_metric_columns(columns: &mut Vec<Column>, rows: &[SummaryRow], config: &SummaryConfig) {
    if !config.show_metrics {
        return;
    }

    let mut maybe_push =
        |key: SummaryParamColumn, header: &'static str, align: Align, values: Vec<Cell>| {
            if !is_hidden(config, key) {
                columns.push(Column {
                    header: header.into(),
                    align,
                    cells: values,
                });
            }
        };

    maybe_push(
        SummaryParamColumn::TrainingTime,
        "time_s",
        Align::Right,
        rows.iter()
            .map(|row| {
                row.metrics
                    .as_ref()
                    .map(|metrics| metrics.training_time.clone())
                    .unwrap_or_else(borrowed_dash)
            })
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::BestEpoch,
        "best_ep",
        Align::Right,
        rows.iter()
            .map(|row| {
                row.metrics
                    .as_ref()
                    .map(|metrics| metrics.best_epoch.clone())
                    .unwrap_or_else(borrowed_dash)
            })
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::FinalEpoch,
        "final_ep",
        Align::Right,
        rows.iter()
            .map(|row| {
                row.metrics
                    .as_ref()
                    .map(|metrics| metrics.final_epoch.clone())
                    .unwrap_or_else(borrowed_dash)
            })
            .collect(),
    );
    maybe_push(
        SummaryParamColumn::MaxError,
        "max_err",
        Align::Right,
        rows.iter()
            .map(|row| {
                row.metrics
                    .as_ref()
                    .map(|metrics| metrics.max_error.clone())
                    .unwrap_or_else(borrowed_dash)
            })
            .collect(),
    );
}

fn format_param_cells(params: &HyperParams) -> ParamCells {
    let scheduler = params.scheduler.to_string();
    let optimizer = params.optimizer.to_string();

    // Real-only regularization; Complex trials show "-" for all five
    // because the Complex model has no dropout / norm / grad-clip /
    // input-noise / weight-decay path (see `ComplexMLPRegressor`
    // docstring — "no normalization, no dropout, no input noise, no
    // gradient clipping — removed by design"). Weight decay is also
    // gated in `HyperParams::optimizer_config` for defense-in-depth;
    // showing "0e0" in the table would misleadingly suggest the
    // optimizer was run with explicit zero decay, when really the
    // whole dimension is skipped by the Complex sampler.
    let is_real = matches!(&params.kind, crate::search_space::ModelKind::Real { .. });
    let (norm_cell, dropout_cell, grad_clip_cell, input_noise_cell) = match &params.kind {
        crate::search_space::ModelKind::Real {
            dropout_p, norm, grad_clip_norm, input_noise_std,
            activation: _,
        } => (
            norm.to_string().into(),
            format_fixed(*dropout_p, 2).into(),
            grad_clip_norm
                .to_f64()
                .map(|value| format_fixed(value, 1).into())
                .unwrap_or_else(borrowed_dash),
            if *input_noise_std > 0.0 {
                format_noise(*input_noise_std).into()
            } else {
                borrowed_dash()
            },
        ),
        crate::search_space::ModelKind::Complex { .. } => (
            borrowed_dash(),
            borrowed_dash(),
            borrowed_dash(),
            borrowed_dash(),
        ),
    };
    let weight_decay_cell: Cell = if is_real {
        format_scientific(params.weight_decay).into()
    } else {
        borrowed_dash()
    };

    ParamCells {
        hidden_size: params.hidden_size.to_string().into(),
        learning_rate: format_scientific(params.lr).into(),
        activation: params.activation_name().to_string().into(),
        optimizer: optimizer.into(),
        scheduler: scheduler.into(),
        batch_size: params.train_batch_size.to_string().into(),
        norm: norm_cell,
        dropout: dropout_cell,
        weight_decay: weight_decay_cell,
        loss: params.loss.to_string().into(),
        grad_clip: grad_clip_cell,
        input_noise: input_noise_cell,
        plateau_factor: if matches!(params.scheduler, SchedulerChoice::Plateau) {
            params
                .scheduler_params
                .plateau_factor
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        plateau_patience: if matches!(params.scheduler, SchedulerChoice::Plateau) {
            params
                .scheduler_params
                .plateau_patience
                .map(|value: i64| value.to_string().into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        plateau_min_lr: if matches!(params.scheduler, SchedulerChoice::Plateau) {
            params
                .scheduler_params
                .plateau_min_lr
                .map(|value| format_scientific(value).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        cosine_eta_min: if matches!(params.scheduler, SchedulerChoice::Cosine) {
            params
                .scheduler_params
                .cosine_eta_min
                .map(|value| format_scientific(value).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        sgd_momentum: if matches!(params.optimizer, OptimizerChoice::SGD) {
            params
                .optimizer_params
                .sgd_momentum
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        sgd_nesterov: if matches!(params.optimizer, OptimizerChoice::SGD) {
            match params.optimizer_params.sgd_nesterov {
                Some(true) => "yes".into(),
                Some(false) => "no".into(),
                None => borrowed_dash(),
            }
        } else {
            borrowed_dash()
        },
        rmsprop_momentum: if matches!(params.optimizer, OptimizerChoice::RMSprop) {
            params
                .optimizer_params
                .rmsprop_momentum
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        rmsprop_alpha: if matches!(params.optimizer, OptimizerChoice::RMSprop) {
            params
                .optimizer_params
                .rmsprop_alpha
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
    }
}

fn format_metric_cells(metrics: &TrialMetrics) -> MetricCells {
    MetricCells {
        training_time: format_fixed(metrics.training_time_secs, 1).into(),
        best_epoch: metrics.best_epoch.to_string().into(),
        final_epoch: metrics.final_epoch.to_string().into(),
        max_error: format_percent(metrics.max_error).into(),
    }
}

fn decode_single_trial_params(trial: &OptimizerCompletedTrial<f64>) -> ParamCells {
    let lookup = SingleTrialLookup::new(trial);

    let hidden_size = lookup
        .int("hidden_size")
        .map(|value| value.to_string())
        .map(Into::into)
        .unwrap_or_else(borrowed_dash);
    let learning_rate = lookup
        .float("lr")
        .map(format_scientific)
        .map(Into::into)
        .unwrap_or_else(borrowed_dash);
    let activation = lookup
        .categorical("activation")
        .map(|index| format_indexed_choice(index, ActivationChoice::from_index).into())
        .unwrap_or_else(borrowed_dash);
    let optimizer_idx = lookup.categorical("optimizer");
    let optimizer = optimizer_idx
        .map(|index| format_indexed_choice(index, OptimizerChoice::from_index).into())
        .unwrap_or_else(borrowed_dash);
    let scheduler_idx = lookup.categorical("scheduler");
    let scheduler = scheduler_idx
        .map(|index| format_indexed_choice(index, SchedulerChoice::from_index).into())
        .unwrap_or_else(borrowed_dash);

    ParamCells {
        hidden_size,
        learning_rate,
        activation,
        optimizer,
        scheduler,
        batch_size: lookup
            .categorical("train_batch_size")
            .map(|index| format_indexed_choice(index, BatchSizeChoice::from_index).into())
            .unwrap_or_else(borrowed_dash),
        norm: lookup
            .categorical("norm")
            .map(|index| format_indexed_choice(index, NormChoice::from_index).into())
            .unwrap_or_else(borrowed_dash),
        dropout: lookup
            .float("dropout_p")
            .map(|value| format_fixed(value, 2))
            .map(Into::into)
            .unwrap_or_else(borrowed_dash),
        weight_decay: lookup
            .float("weight_decay")
            .map(format_scientific)
            .map(Into::into)
            .unwrap_or_else(borrowed_dash),
        loss: lookup
            .categorical("loss")
            .map(|index| format_indexed_choice(index, LossChoice::from_index).into())
            .unwrap_or_else(borrowed_dash),
        grad_clip: lookup
            .categorical("grad_clip_norm")
            .map(format_clip_index)
            .map(Into::into)
            .unwrap_or_else(borrowed_dash),
        input_noise: lookup
            .float("input_noise_std")
            .map(|value| {
                if value > 0.0 {
                    format_noise(value).into()
                } else {
                    borrowed_dash()
                }
            })
            .unwrap_or_else(borrowed_dash),
        plateau_factor: if matches!(scheduler_idx, Some(1)) {
            lookup
                .float("plateau_factor")
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        plateau_patience: if matches!(scheduler_idx, Some(1)) {
            lookup
                .int("plateau_patience")
                .map(|value| value.to_string().into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        plateau_min_lr: if matches!(scheduler_idx, Some(1)) {
            lookup
                .float("plateau_min_lr")
                .map(|value| format_scientific(value).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        cosine_eta_min: if matches!(scheduler_idx, Some(2)) {
            lookup
                .float("cosine_eta_min")
                .map(|value| format_scientific(value).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        sgd_momentum: if matches!(optimizer_idx, Some(3)) {
            lookup
                .float("sgd_momentum")
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        sgd_nesterov: if matches!(optimizer_idx, Some(3)) {
            lookup
                .categorical("sgd_nesterov")
                .map(|value| if value == 0 { "no" } else { "yes" })
                .map(Into::into)
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        rmsprop_momentum: if matches!(optimizer_idx, Some(2)) {
            lookup
                .float("rmsprop_momentum")
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
        rmsprop_alpha: if matches!(optimizer_idx, Some(2)) {
            lookup
                .float("rmsprop_alpha")
                .map(|value| format_fixed(value, 2).into())
                .unwrap_or_else(borrowed_dash)
        } else {
            borrowed_dash()
        },
    }
}

fn is_hidden(config: &SummaryConfig, column: SummaryParamColumn) -> bool {
    config.force_hide_params.contains(&column)
}

// ---------------------------------------------------------------------------
// Multi-objective summary
// ---------------------------------------------------------------------------

fn format_multi_objective_summary(
    meta: SummaryMeta<'_>,
    results: &MultiObjectiveResults,
    config: &SummaryConfig,
) -> String {
    let symbols = Symbols::new(config.use_unicode);
    let pareto_front = results.pareto_front();
    let mut pareto_ids = HashSet::with_capacity(pareto_front.len());
    for trial in pareto_front {
        pareto_ids.insert(trial.trial_number);
    }

    let mut trials: Vec<_> = results.trials.iter().collect();
    let display_count = config.top_n.min(trials.len());
    if display_count > 0 && display_count < trials.len() {
        trials.select_nth_unstable_by(display_count - 1, |left, right| {
            compare_multi_trials(left, right, results.directions)
        });
        trials.truncate(display_count);
    }
    trials.sort_by(|left, right| compare_multi_trials(left, right, results.directions));

    let rows: Vec<_> = trials
        .into_iter()
        .map(|trial| SummaryRow {
            trial_id: trial.trial_number.to_string().into(),
            pareto: if pareto_ids.contains(&trial.trial_number) {
                symbols.pareto.into()
            } else {
                "".into()
            },
            objective0: format_percent(trial.objectives[0]).into(),
            objective1: Some(format_param_count(trial.objectives[1]).into()),
            status: format_multi_status(trial.status, trial.constraint_value, &symbols).into(),
            params: format_param_cells(&trial.params),
            metrics: config
                .show_metrics
                .then(|| format_metric_cells(&trial.metrics)),
        })
        .collect();

    let completed = results
        .trials
        .iter()
        .filter(|trial| trial.status == TrialStatus::Completed)
        .count();
    let failed = results.trials.len().saturating_sub(completed);
    let header = format!(
        "Study: {} | Trials: {} completed, {} failed | Constraint: {}",
        meta.study_name,
        completed,
        failed,
        format_constraint(results.constraint_threshold, symbols.less_equal),
    );
    let directions = format!(
        "Directions: [{} OK@1%, {} hidden_size]",
        format_direction(results.directions[0]),
        format_direction(results.directions[1]),
    );
    let footer = format!(
        "{} = Pareto front ({} trials) | Best compromise: {}{}",
        symbols.pareto,
        pareto_ids.len(),
        format_best_compromise(results),
        format_total_duration(meta.total_duration),
    );

    render_summary(header, directions, build_multi_columns(&rows, config), footer)
}

// ---------------------------------------------------------------------------
// Single-objective summary
// ---------------------------------------------------------------------------

pub(crate) fn format_single_objective_summary(
    meta: SummaryMeta<'_>,
    direction: Direction,
    trials: &[OptimizerCompletedTrial<f64>],
    pruned_count: usize,
    config: &SummaryConfig,
) -> String {
    let symbols = Symbols::new(config.use_unicode);
    let mut sorted = trials.to_vec();
    let display_count = config.top_n.min(sorted.len());
    if display_count > 0 && display_count < sorted.len() {
        sorted.select_nth_unstable_by(display_count - 1, |left, right| {
            compare_single_trials(left, right, direction)
        });
        sorted.truncate(display_count);
    }
    sorted.sort_by(|left, right| compare_single_trials(left, right, direction));

    let rows: Vec<_> = sorted
        .iter()
        .map(|trial| SummaryRow {
            trial_id: trial.id.to_string().into(),
            pareto: "".into(),
            objective0: format_generic_value(trial.value).into(),
            objective1: None,
            status: format_single_status(trial, &symbols).into(),
            params: decode_single_trial_params(trial),
            metrics: config.show_metrics.then(|| MetricCells {
                training_time: borrowed_dash(),
                best_epoch: borrowed_dash(),
                final_epoch: borrowed_dash(),
                max_error: borrowed_dash(),
            }),
        })
        .collect();

    let completed = sorted
        .iter()
        .filter(|trial| trial.state == TrialState::Complete)
        .count();
    let header = format!(
        "Study: {} | Trials: {} completed, {} pruned",
        meta.study_name, completed, pruned_count,
    );
    let directions = format!("Direction: {} value", format_direction(direction));
    let footer = format!(
        "Best trial: {}{}",
        format_best_single_trial(&sorted, direction),
        format_total_duration(meta.total_duration),
    );

    render_summary(header, directions, build_single_columns(&rows, config), footer)
}

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

fn format_multi_status(status: TrialStatus, constraint_value: f64, symbols: &Symbols<'_>) -> String {
    match status {
        TrialStatus::Completed if constraint_value <= 0.0 => symbols.ok.to_string(),
        TrialStatus::Completed => symbols.infeasible.to_string(),
        TrialStatus::Failed => symbols.error.to_string(),
    }
}

fn format_single_status(trial: &OptimizerCompletedTrial<f64>, symbols: &Symbols<'_>) -> String {
    match trial.state {
        TrialState::Complete if is_constraint_feasible(&trial.constraints) => {
            symbols.ok.to_string()
        }
        TrialState::Complete => symbols.infeasible.to_string(),
        TrialState::Pruned => symbols.pruned.to_string(),
        TrialState::Failed => symbols.error.to_string(),
        TrialState::Running => String::from("run"),
    }
}

// ---------------------------------------------------------------------------
// Sorting helpers
// ---------------------------------------------------------------------------

fn compare_multi_trials(
    left: &CompletedTrial,
    right: &CompletedTrial,
    directions: [Direction; 3],
) -> Ordering {
    trial_bucket(
        left.status == TrialStatus::Completed,
        left.is_feasible(),
    )
    .cmp(&trial_bucket(
        right.status == TrialStatus::Completed,
        right.is_feasible(),
    ))
    .then_with(|| compare_value(left.objectives[0], right.objectives[0], directions[0]))
    .then_with(|| compare_value(left.objectives[1], right.objectives[1], directions[1]))
    .then_with(|| compare_value(left.objectives[2], right.objectives[2], directions[2]))
    .then_with(|| left.trial_number.cmp(&right.trial_number))
}

fn compare_single_trials(
    left: &OptimizerCompletedTrial<f64>,
    right: &OptimizerCompletedTrial<f64>,
    direction: Direction,
) -> Ordering {
    single_trial_bucket(left)
        .cmp(&single_trial_bucket(right))
        .then_with(|| compare_value(left.value, right.value, direction))
        .then_with(|| left.id.cmp(&right.id))
}

fn single_trial_bucket(trial: &OptimizerCompletedTrial<f64>) -> u8 {
    match trial.state {
        TrialState::Complete if is_constraint_feasible(&trial.constraints) => 0,
        TrialState::Complete => 1,
        TrialState::Pruned => 2,
        TrialState::Failed => 3,
        TrialState::Running => 4,
    }
}

// ---------------------------------------------------------------------------
// Footer helpers
// ---------------------------------------------------------------------------

fn format_best_compromise(results: &MultiObjectiveResults) -> String {
    results
        .best_compromise()
        .map(|trial| {
            format!(
                "trial {} (OK@1%={}, params={}, max_err={:.2}%)",
                trial.trial_number,
                format_percent(trial.objectives[0]),
                format_param_count(trial.objectives[1]),
                trial.objectives[2],
            )
        })
        .unwrap_or_else(|| String::from("none"))
}

fn format_best_single_trial(
    trials: &[OptimizerCompletedTrial<f64>],
    direction: Direction,
) -> String {
    let best = trials
        .iter()
        .filter(|trial| trial.state == TrialState::Complete)
        .min_by(|left, right| compare_single_trials(left, right, direction));

    best.map(|trial| {
        format!(
            "trial {} (value={})",
            trial.id,
            format_generic_value(trial.value)
        )
    })
    .unwrap_or_else(|| String::from("none"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use optimizer::distribution::{
        CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    };
    use optimizer::parameter::{ParamId, ParamValue};

    use super::*;
    use crate::search_space::{
        BatchSizeChoice, LossChoice, NormChoice,
        OptimizerChoice, OptimizerHyperParams, SchedulerChoice,
        SchedulerHyperParams,
    };
    use crate::pareto::{CompletedTrial, MultiObjectiveResults};
    use crate::evaluation::TrialMetrics;

    fn sample_params(hidden_size: i64) -> HyperParams {
        HyperParams {
            hidden_size,
            optimizer: OptimizerChoice::AdamW,
            lr: 1.2e-3,
            train_batch_size: BatchSizeChoice::B64,
            weight_decay: 1.0e-5,
            loss: LossChoice::Mse,
            loss_params: crate::search_space::LossHyperParams::default(),
            scheduler: SchedulerChoice::Cosine,
            scheduler_params: SchedulerHyperParams {
                cosine_eta_min: Some(1.0e-5),
                ..Default::default()
            },
            optimizer_params: OptimizerHyperParams::default(),
            kind: crate::search_space::ModelKind::Real {
                activation: sparam_models::Activation::GELU,
                dropout_p: 0.10,
                norm: NormChoice::LayerNorm,
                grad_clip_norm: crate::search_space::GradClipChoice::Clip10,
                input_noise_std: 0.005,
            },
        }
    }

    fn trial_metrics(ok_at_1pct: f64, max_error: f64) -> TrialMetrics {
        TrialMetrics {
            best_val_loss: 0.001,
            param_count: 320,
            stopped_early: false,
            best_epoch: 10,
            final_epoch: 20,
            training_time_secs: 1.5,
            ok_at_1pct,
            max_error,
        }
    }

    fn multi_results() -> MultiObjectiveResults {
        use crate::search_space::{GradClipChoice, ModelKind};

        let mut plateau = sample_params(16);
        plateau.optimizer = OptimizerChoice::Adam;
        plateau.lr = 8.5e-4;
        plateau.train_batch_size = BatchSizeChoice::B32;
        plateau.weight_decay = 1.0e-6;
        plateau.scheduler = SchedulerChoice::Plateau;
        plateau.scheduler_params = SchedulerHyperParams {
            plateau_factor: Some(0.5),
            plateau_patience: Some(5),
            plateau_min_lr: Some(1.0e-5),
            ..Default::default()
        };
        plateau.kind = ModelKind::Real {
            activation: sparam_models::Activation::ReLU,
            dropout_p: 0.05,
            norm: NormChoice::None,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        };

        let mut sgd = sample_params(64);
        sgd.optimizer = OptimizerChoice::SGD;
        sgd.lr = 2.1e-3;
        sgd.train_batch_size = BatchSizeChoice::B128;
        sgd.weight_decay = 1.0e-4;
        sgd.loss = LossChoice::SmoothL1;
        sgd.scheduler = SchedulerChoice::None;
        sgd.scheduler_params = SchedulerHyperParams::default();
        sgd.optimizer_params = OptimizerHyperParams {
            sgd_momentum: Some(0.9),
            sgd_nesterov: Some(true),
            ..Default::default()
        };
        sgd.kind = ModelKind::Real {
            activation: sparam_models::Activation::GELU,
            dropout_p: 0.20,
            norm: NormChoice::BatchNorm,
            grad_clip_norm: GradClipChoice::None,
            input_noise_std: 0.0,
        };

        let mut rms = sample_params(32);
        rms.optimizer = OptimizerChoice::RMSprop;
        rms.lr = 1.8e-3;
        rms.scheduler = SchedulerChoice::Cosine;
        rms.scheduler_params = SchedulerHyperParams {
            cosine_eta_min: Some(1.0e-6),
            ..Default::default()
        };
        rms.optimizer_params = OptimizerHyperParams {
            rmsprop_momentum: Some(0.1),
            rmsprop_alpha: Some(0.95),
            ..Default::default()
        };
        rms.kind = ModelKind::Real {
            activation: sparam_models::Activation::Tanh,
            dropout_p: 0.15,
            norm: NormChoice::LayerNorm,
            grad_clip_norm: GradClipChoice::Clip20,
            input_noise_std: 0.0,
        };

        MultiObjectiveResults {
            trials: vec![
                CompletedTrial {
                    trial_number: 42,
                    params: sample_params(32),
                    objectives: [98.5, 32.0, 1.0],
                    num_objectives: 3,
                    metrics: trial_metrics(98.5, 5.0),
                    constraint_value: -5.0,
                    status: TrialStatus::Completed,
                },
                CompletedTrial {
                    trial_number: 67,
                    params: plateau,
                    objectives: [98.2, 16.0, 1.2],
                    num_objectives: 3,
                    metrics: trial_metrics(98.2, 4.0),
                    constraint_value: -6.0,
                    status: TrialStatus::Completed,
                },
                CompletedTrial {
                    trial_number: 31,
                    params: sgd,
                    objectives: [97.8, 64.0, 1.8],
                    num_objectives: 3,
                    metrics: trial_metrics(97.8, 4.5),
                    constraint_value: -5.5,
                    status: TrialStatus::Completed,
                },
                CompletedTrial {
                    trial_number: 55,
                    params: rms,
                    objectives: [97.5, 32.0, 3.0],
                    num_objectives: 3,
                    metrics: trial_metrics(97.5, 12.0),
                    constraint_value: 2.0,
                    status: TrialStatus::Completed,
                },
                CompletedTrial {
                    trial_number: 12,
                    params: sample_params(32),
                    objectives: [f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY],
                    num_objectives: 3,
                    metrics: TrialMetrics::default(),
                    constraint_value: f64::INFINITY,
                    status: TrialStatus::Failed,
                },
            ],
            directions: [Direction::Maximize, Direction::Minimize, Direction::Minimize],
            constraint_threshold: Some(10.0),
        }
    }

    fn insert_param(
        params: &mut HashMap<ParamId, ParamValue>,
        labels: &mut HashMap<ParamId, String>,
        distributions: &mut HashMap<ParamId, Distribution>,
        label: &str,
        value: ParamValue,
        distribution: Distribution,
    ) {
        let id = ParamId::new();
        params.insert(id, value);
        labels.insert(id, label.to_string());
        distributions.insert(id, distribution);
    }

    fn single_trial(id: u64, value: f64, state: TrialState) -> OptimizerCompletedTrial<f64> {
        let mut params = HashMap::new();
        let mut labels = HashMap::new();
        let mut distributions = HashMap::new();

        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "hidden_size",
            ParamValue::Int(32),
            Distribution::Int(IntDistribution {
                low: 8,
                high: 64,
                log_scale: false,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "activation",
            ParamValue::Categorical(2),
            Distribution::Categorical(CategoricalDistribution { n_choices: 5 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "optimizer",
            ParamValue::Categorical(3),
            Distribution::Categorical(CategoricalDistribution { n_choices: 4 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "lr",
            ParamValue::Float(2.0e-3),
            Distribution::Float(FloatDistribution {
                low: 1.0e-4,
                high: 5.0e-2,
                log_scale: true,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "train_batch_size",
            ParamValue::Categorical(3),
            Distribution::Categorical(CategoricalDistribution { n_choices: 8 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "weight_decay",
            ParamValue::Float(1.0e-4),
            Distribution::Float(FloatDistribution {
                low: 1.0e-10,
                high: 1.0e-2,
                log_scale: true,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "dropout_p",
            ParamValue::Float(0.2),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 0.4,
                log_scale: false,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "norm",
            ParamValue::Categorical(2),
            Distribution::Categorical(CategoricalDistribution { n_choices: 3 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "loss",
            ParamValue::Categorical(1),
            Distribution::Categorical(CategoricalDistribution { n_choices: 3 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "grad_clip_norm",
            ParamValue::Categorical(0),
            Distribution::Categorical(CategoricalDistribution { n_choices: 5 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "input_noise_std",
            ParamValue::Float(0.0),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 0.01,
                log_scale: false,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "scheduler",
            ParamValue::Categorical(0),
            Distribution::Categorical(CategoricalDistribution { n_choices: 3 }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "sgd_momentum",
            ParamValue::Float(0.9),
            Distribution::Float(FloatDistribution {
                low: 0.0,
                high: 0.95,
                log_scale: false,
                step: None,
            }),
        );
        insert_param(
            &mut params,
            &mut labels,
            &mut distributions,
            "sgd_nesterov",
            ParamValue::Categorical(1),
            Distribution::Categorical(CategoricalDistribution { n_choices: 2 }),
        );

        OptimizerCompletedTrial {
            id,
            params,
            distributions,
            param_labels: labels,
            value,
            intermediate_values: Vec::new(),
            state,
            user_attrs: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    #[test]
    fn multi_summary_marks_pareto_and_preserves_full_columns() {
        let summary = format_summary(
            SummaryInput::MultiObjective {
                meta: SummaryMeta::new("mlp_hpo_v1").with_total_duration(Duration::from_secs(12)),
                results: &multi_results(),
            },
            &SummaryConfig::default(),
        );

        assert!(summary.contains("Study: mlp_hpo_v1"));
        assert!(summary.contains("Directions: [Maximize OK@1%, Minimize hidden_size]"));
        assert!(summary.contains("plat_f"));
        assert!(summary.contains("sgd_m"));
        assert!(summary.contains("rms_α"));
        assert!(summary.contains("★ = Pareto front"));
        assert!(summary.contains("trial 42"));
        assert!(summary.contains("Best compromise: trial 42"));
        assert!(summary.contains("Duration: 12.0s"));
    }

    #[test]
    fn multi_summary_ascii_fallback_uses_ascii_markers() {
        let summary = format_summary(
            SummaryInput::MultiObjective {
                meta: SummaryMeta::new("ascii"),
                results: &multi_results(),
            },
            &SummaryConfig {
                use_unicode: false,
                ..SummaryConfig::default()
            },
        );

        assert!(summary.contains("* = Pareto front"));
        assert!(summary.contains("[ok]"));
        assert!(summary.contains("[lim]"));
        assert!(summary.contains("[err]"));
    }

    #[test]
    fn single_summary_decodes_known_param_labels() {
        let trials = vec![single_trial(7, 0.125, TrialState::Complete)];
        let summary = format_single_objective_summary(
            SummaryMeta::new("single_hpo"),
            Direction::Minimize,
            &trials,
            0,
            &SummaryConfig::default(),
        );

        assert!(summary.contains("Study: single_hpo | Trials: 1 completed, 0 pruned"));
        assert!(summary.contains("Direction: Minimize value"));
        assert!(summary.contains("gelu"));
        assert!(summary.contains("sgd"));
        assert!(summary.contains("btch"));
        assert!(summary.contains("smth"));
        assert!(summary.contains("yes"));
    }

    #[test]
    fn summary_can_hide_columns_explicitly() {
        let summary = format_summary(
            SummaryInput::MultiObjective {
                meta: SummaryMeta::new("hidden_cols"),
                results: &multi_results(),
            },
            &SummaryConfig {
                force_hide_params: vec![
                    SummaryParamColumn::PlateauFactor,
                    SummaryParamColumn::RmspropAlpha,
                ],
                ..SummaryConfig::default()
            },
        );

        assert!(!summary.contains("plat_f"));
        assert!(!summary.contains("rms_α"));
        assert!(summary.contains("sgd_m"));
    }

    #[test]
    fn empty_multi_summary_is_handled() {
        let results = MultiObjectiveResults {
            trials: Vec::new(),
            directions: [Direction::Maximize, Direction::Minimize, Direction::Minimize],
            constraint_threshold: None,
        };
        let summary = format_summary(
            SummaryInput::MultiObjective {
                meta: SummaryMeta::new("empty"),
                results: &results,
            },
            &SummaryConfig::default(),
        );

        assert!(summary.contains("No trials available."));
        assert!(summary.contains("Best compromise: none"));
    }
}
