//! Non-blocking channel-based logger for training output.
//!
//! All console output during training is dispatched through a bounded
//! `mpsc::sync_channel` and printed on a dedicated background thread so that
//! the main training loop is never stalled by I/O syscalls.
//!
//! Each log line can carry two optional prefixes so concurrent workloads
//! (two HPO studies, two training runs) don't produce indistinguishable
//! interleaved stderr output:
//! - **`context`** — a long-lived tag like `"study_a"` or `"run-42"` set
//!   via [`LogSender::with_context`]. Identifies the top-level workflow
//!   so two concurrent studies' lines stay visually separated.
//! - **`trial_id`** — per-trial tag set via [`LogSender::with_trial_id`]
//!   for HPO trials.

use std::io::Write;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

/// Structured log message emitted by the training loop and workflow layer.
pub enum LogMessage {
    /// Per-epoch progress line emitted by the trainer.
    Progress {
        epoch: usize,
        total: usize,
        loss: f64,
        val_loss: Option<f64>,
        lr: f64,
        elapsed_secs: f64,
        /// `Some((bad_epochs, patience))` when early-stopping is active.
        patience_status: Option<(usize, usize)>,
        /// `Some((current_epoch, warmup_epochs))` during warmup phase.
        warmup_status: Option<(usize, usize)>,
    },
    /// General informational message.
    Info(String),
    /// Warning message.
    Warning(String),
    /// Error message.
    Error(String),
    /// Visual separator line.
    Separator,
    /// Key-value metric row.
    MetricRow { label: String, value: String },
}

struct Envelope {
    context: Option<Arc<str>>,
    trial_id: Option<usize>,
    msg: LogMessage,
}

/// Cheap-to-clone handle that sends structured messages to the background
/// logger thread via a bounded channel.
///
/// When the inner sender is `None` (constructed via [`LogSender::null()`]),
/// all messages are silently discarded with no thread allocation.
///
/// The `context` tag (set via [`LogSender::with_context`]) is the primary
/// way to keep two concurrent studies' log output distinguishable on a
/// shared stderr — every line is prefixed with `[<context>]` so
/// interleaved lines stay attributable.
#[derive(Clone)]
pub struct LogSender {
    tx: Option<mpsc::SyncSender<Envelope>>,
    context: Option<Arc<str>>,
    trial_id: Option<usize>,
}

impl LogSender {
    /// Create a new active logger.
    ///
    /// Returns a `(LogSender, LogWorker)` pair.  The `LogWorker` **must** be
    /// kept alive until all output has been flushed — dropping it joins the
    /// background thread.
    pub fn new() -> (Self, LogWorker) {
        let (tx, rx) = mpsc::sync_channel::<Envelope>(512);
        let handle = thread::Builder::new()
            .name("log-worker".into())
            .spawn(move || {
                let stderr = std::io::stderr();
                for envelope in rx {
                    let mut out = stderr.lock();
                    render(&mut out, envelope.context.as_deref(), envelope.trial_id, &envelope.msg);
                    let _ = out.flush();
                }
            })
            .expect("failed to spawn log-worker thread");
        (
            LogSender {
                tx: Some(tx),
                context: None,
                trial_id: None,
            },
            LogWorker(Some(handle)),
        )
    }

    /// No-op sink that discards all messages.  No thread is allocated.
    #[must_use]
    pub fn null() -> Self {
        LogSender {
            tx: None,
            context: None,
            trial_id: None,
        }
    }

    /// Consume this sender and return one with a `context` tag stamped
    /// onto every future message. Used to keep two concurrent HPO
    /// studies / training runs visually distinguishable on a shared
    /// stderr — every line the returned sender produces is prefixed
    /// with `[<context>]`. The tag is stored as `Arc<str>` so
    /// subsequent sends only do a reference-count bump, not a string
    /// copy.
    ///
    /// **Takes `self` by value deliberately**: previously this method
    /// was `&self` and callers paired it with a bound `LogWorker`:
    /// ```ignore
    /// let (log_sender, worker) = LogSender::new();
    /// let _worker = worker;
    /// let log = log_sender.with_context("study");  // log_sender lives on!
    /// ```
    /// That left `log_sender` alive in the enclosing scope, so the
    /// mpsc channel stayed open when `_worker` dropped → `join()`
    /// hung forever waiting for the worker thread (which itself was
    /// waiting for the channel to close). Consuming `self` prevents
    /// the stale-sender footgun: the only way to keep a pre-context
    /// copy around is an explicit `self.clone().with_context(...)`.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<Arc<str>>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Return a clone of this sender tagged with a trial ID. When
    /// combined with [`with_context`](Self::with_context), the line
    /// prefix is `[<context>:trial-<id>]`.
    #[must_use]
    pub fn with_trial_id(&self, id: usize) -> Self {
        LogSender {
            tx: self.tx.clone(),
            context: self.context.clone(),
            trial_id: Some(id),
        }
    }

    /// Send a log message.  Returns immediately without blocking the caller
    /// unless the bounded channel buffer (512 slots) is full, in which case
    /// the message is silently dropped rather than stalling the training loop.
    pub fn send(&self, msg: LogMessage) {
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(Envelope {
                context: self.context.clone(),
                trial_id: self.trial_id,
                msg,
            });
        }
    }

    /// Returns `true` when this sender is connected to a real worker thread.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.tx.is_some()
    }
}

/// Handle to the background logger thread.
///
/// Dropping the `LogWorker` (after all `LogSender` clones have been dropped)
/// drains any remaining messages in the channel, flushes stderr, and joins the
/// thread — no messages are lost on exit.
pub struct LogWorker(Option<thread::JoinHandle<()>>);

impl Drop for LogWorker {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let _ = handle.join();
        }
    }
}

/// Compute the `[context:trial-N] ` / `[context] ` / `[trial-N] ` / empty
/// prefix used by every `render` arm. Centralized so all arms stay in
/// sync when a new prefix dimension is added.
fn line_prefix(context: Option<&str>, trial_id: Option<usize>) -> String {
    match (context, trial_id) {
        (Some(ctx), Some(tid)) => format!("[{ctx}:trial-{tid}] "),
        (Some(ctx), None) => format!("[{ctx}] "),
        (None, Some(tid)) => format!("[trial-{tid}] "),
        (None, None) => String::new(),
    }
}

fn render(out: &mut impl Write, context: Option<&str>, trial_id: Option<usize>, msg: &LogMessage) {
    let prefix = line_prefix(context, trial_id);
    match msg {
        LogMessage::Progress {
            epoch,
            total,
            loss,
            val_loss,
            lr,
            elapsed_secs,
            patience_status,
            warmup_status,
        } => {
            let val_display = match val_loss {
                Some(v) => format!("{v:.6}"),
                None => "skipped".to_string(),
            };
            let status = if let Some((bad, patience)) = patience_status {
                format!(" patience={bad}/{patience}")
            } else if let Some((cur, warmup)) = warmup_status {
                format!(" warmup={cur}/{warmup}")
            } else {
                String::new()
            };
            let _ = writeln!(
                out,
                "{prefix}[Epoch {epoch:>3}/{total}] train_loss={loss:.6} val_loss={val_display} lr={lr:.6}{status} ({elapsed_secs:.1}s)",
            );
        }
        LogMessage::Info(s) => {
            let _ = writeln!(out, "{prefix}{s}");
        }
        LogMessage::Warning(s) => {
            let _ = writeln!(out, "{prefix}Warning: {s}");
        }
        LogMessage::Error(s) => {
            let _ = writeln!(out, "{prefix}Error: {s}");
        }
        LogMessage::Separator => {
            let _ = writeln!(out, "{prefix}{}", "-".repeat(60));
        }
        LogMessage::MetricRow { label, value } => {
            let _ = writeln!(out, "{prefix}  {label}: {value}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sender_discards_messages() {
        let log = LogSender::null();
        assert!(!log.is_active());
        log.send(LogMessage::Info("hello".into()));
    }

    #[test]
    fn active_sender_delivers_messages() {
        let (log, worker) = LogSender::new();
        assert!(log.is_active());
        log.send(LogMessage::Info("test message".into()));
        log.send(LogMessage::Progress {
            epoch: 1,
            total: 10,
            loss: 0.123456,
            val_loss: Some(0.654321),
            lr: 0.001,
            elapsed_secs: 1.2,
            patience_status: None,
            warmup_status: None,
        });
        drop(log);
        drop(worker);
    }

    #[test]
    fn clone_shares_channel() {
        let (log, worker) = LogSender::new();
        let log2 = log.clone();
        log.send(LogMessage::Info("from original".into()));
        log2.send(LogMessage::Info("from clone".into()));
        drop(log);
        drop(log2);
        drop(worker);
    }

    #[test]
    fn with_trial_id_prefixes_output() {
        let (log, worker) = LogSender::new();
        let trial_log = log.with_trial_id(42);
        trial_log.send(LogMessage::Info("trial output".into()));
        drop(trial_log);
        drop(log);
        drop(worker);
    }

    #[test]
    fn render_progress_without_trial() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            None,
            &LogMessage::Progress {
                epoch: 1,
                total: 100,
                loss: 0.123456,
                val_loss: Some(0.654321),
                lr: 0.001000,
                elapsed_secs: 1.2,
                patience_status: None,
                warmup_status: None,
            },
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("[Epoch   1/100]"));
        assert!(output.contains("train_loss=0.123456"));
        assert!(output.contains("val_loss=0.654321"));
        assert!(output.contains("lr=0.001000"));
        assert!(output.contains("(1.2s)"));
    }

    #[test]
    fn render_progress_with_patience() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            None,
            &LogMessage::Progress {
                epoch: 10,
                total: 100,
                loss: 0.1,
                val_loss: Some(0.2),
                lr: 0.001,
                elapsed_secs: 0.5,
                patience_status: Some((3, 10)),
                warmup_status: None,
            },
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("patience=3/10"));
    }

    #[test]
    fn render_progress_with_warmup() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            None,
            &LogMessage::Progress {
                epoch: 2,
                total: 100,
                loss: 0.1,
                val_loss: Some(0.2),
                lr: 0.001,
                elapsed_secs: 0.5,
                patience_status: None,
                warmup_status: Some((2, 5)),
            },
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("warmup=2/5"));
    }

    #[test]
    fn render_progress_val_loss_skipped() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            None,
            &LogMessage::Progress {
                epoch: 1,
                total: 10,
                loss: 0.5,
                val_loss: None,
                lr: 0.01,
                elapsed_secs: 0.3,
                patience_status: None,
                warmup_status: None,
            },
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("val_loss=skipped"));
    }

    #[test]
    fn render_with_trial_prefix() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            Some(7),
            &LogMessage::Info("hello".into()),
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.starts_with("[trial-7] hello"));
    }

    /// Context prefix alone — used when two concurrent HPO studies
    /// log to the same stderr so interleaved lines stay attributable
    /// to their study.
    #[test]
    fn render_with_context_prefix() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            Some("study_a"),
            None,
            &LogMessage::Info("trial 5/50 complete".into()),
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.starts_with("[study_a] trial 5/50 complete"),
            "got: {output}"
        );
    }

    /// Context + trial_id combined — the per-trial progress lines
    /// emitted inside an HPO evaluator (which set `with_trial_id`)
    /// still carry the outer study context.
    #[test]
    fn render_with_context_and_trial_prefix_combined() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            Some("study_b"),
            Some(42),
            &LogMessage::Info("epoch 10 done".into()),
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.starts_with("[study_b:trial-42] epoch 10 done"),
            "got: {output}"
        );
    }

    /// End-to-end: tagging a sender with `with_context` (which
    /// consumes the sender) and sending via the bounded channel must
    /// deliver the prefix to the background worker's render output.
    /// Consuming `self` is what prevents the drop-order deadlock in
    /// callers — see the doc-comment on [`LogSender::with_context`]
    /// for the history.
    #[test]
    fn with_context_prefixes_sent_messages() {
        let (log, worker) = LogSender::new();
        let tagged = log.with_context("study_c");
        tagged.send(LogMessage::Info("hello".into()));
        drop(tagged);
        // No `drop(log)` here — `log` was consumed by `with_context`.
        drop(worker);
    }

    #[test]
    fn render_metric_row() {
        let mut buf = Vec::new();
        render(
            &mut buf,
            None,
            None,
            &LogMessage::MetricRow {
                label: "R\u{b2}".into(),
                value: "0.99500".into(),
            },
        );
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("  R\u{b2}: 0.99500"));
    }

    #[test]
    fn render_separator() {
        let mut buf = Vec::new();
        render(&mut buf, None, None, &LogMessage::Separator);
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("----"));
    }

    #[test]
    fn render_warning() {
        let mut buf = Vec::new();
        render(&mut buf, None, None, &LogMessage::Warning("low lr".into()));
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Warning: low lr"));
    }

    #[test]
    fn render_error() {
        let mut buf = Vec::new();
        render(&mut buf, None, None, &LogMessage::Error("diverged".into()));
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("Error: diverged"));
    }
}
