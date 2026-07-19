//! Mini-batch data loader with optional shuffling.
//!
//! [`DataLoader`] wraps input/target tensor pairs and yields batches for
//! training and evaluation. Supports full-batch and fixed-size modes plus
//! deterministic shuffling via a per-epoch seed.
//!
//! **Scaling note:** the loader does NOT apply feature/target scalers
//! per batch. Callers fit a [`Scaler`](crate::scaling::Scaler) once,
//! transform the full train/val/test tensors up front, and hand
//! already-scaled tensors to [`DataLoader::new`]. This removes one
//! element-wise subtract + one element-wise divide from every
//! per-batch iteration — at small batch sizes the per-batch allocation
//! cost was dwarfing the actual matmul work.

use std::{
    cell::{Cell, RefCell},
    iter::FusedIterator,
};

use candle_core::{Result, Tensor};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

use sparam_core::error::candle_msg;
use sparam_core::rng::get_global_seed;

/// Batch sizing mode for [`DataLoader`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchSize {
    /// Yield batches with a fixed maximum size.
    Fixed(usize),
    /// Yield the entire dataset as a single batch.
    All,
}

impl BatchSize {
    fn validate(self) -> Result<()> {
        if matches!(self, Self::Fixed(0)) {
            return Err(candle_msg(
                "DataLoader batch size must be greater than zero",
            ));
        }

        Ok(())
    }

    fn resolve(self, sample_count: usize) -> usize {
        match self {
            Self::Fixed(batch_size) => batch_size,
            Self::All => sample_count,
        }
    }
}

/// Mini-batch tensor loader for supervised learning datasets.
///
/// The loader keeps full feature and target tensors in memory and yields batched
/// tensors through [`DataLoader::iter`]. When shuffling is enabled, each iterator
/// creation derives a deterministic epoch-specific permutation from the configured
/// seed without mutating the underlying dataset tensors.
pub struct DataLoader {
    features: Tensor,
    targets: Tensor,
    batch_size: BatchSize,
    shuffle: bool,
    shuffle_seed: u64,
    epoch: Cell<u64>,
    shuffle_indices_scratch: RefCell<Vec<u32>>,
}

impl DataLoader {
    /// Build a loader from separate feature and target tensors.
    ///
    /// Tensors are consumed as-is — apply any scaling (e.g.
    /// [`StandardScaler::transform`](crate::scaling::StandardScaler::transform))
    /// before calling this constructor, not per batch.
    ///
    /// The default shuffle seed is captured here via
    /// [`sparam_core::rng::get_global_seed`]; a later change to the
    /// global seed does not affect an already-constructed loader. Use
    /// [`with_seed`](Self::with_seed) to override the seed explicitly.
    pub fn new(features: Tensor, targets: Tensor, batch_size: BatchSize) -> Result<Self> {
        validate_dataset_pair(&features, &targets)?;
        batch_size.validate()?;

        Ok(Self {
            features,
            targets,
            batch_size,
            shuffle: false,
            shuffle_seed: get_global_seed(),
            epoch: Cell::new(0),
            shuffle_indices_scratch: RefCell::new(Vec::new()),
        })
    }

    /// Build a loader by splitting one supervised dataset tensor along the feature axis.
    ///
    /// `feature_count` defines how many leading columns belong to inputs; the remaining
    /// columns are treated as targets.
    ///
    /// The split materialises fresh contiguous tensors (one copy per
    /// half), not `narrow` views — required because shuffled batching
    /// uses `index_select`, which needs contiguous inputs. One-time
    /// copy at construction; zero per-batch cost.
    pub fn from_dataset_columns(
        data: Tensor,
        feature_count: usize,
        batch_size: BatchSize,
    ) -> Result<Self> {
        let (_, total_columns) =
            validate_matrix(&data, "DataLoader::from_dataset_columns", "data")?;
        if feature_count == 0 || feature_count >= total_columns {
            return Err(candle_msg(format!(
                "DataLoader::from_dataset_columns expects feature_count in 1..{total_columns}, got {feature_count}"
            )));
        }

        let features = data.narrow(1, 0, feature_count)?.contiguous()?;
        let targets = data
            .narrow(1, feature_count, total_columns - feature_count)?
            .contiguous()?;
        Self::new(features, targets, batch_size)
    }

    /// Enable or disable shuffling before each new iterator/epoch.
    #[must_use]
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the deterministic shuffle seed used for epoch permutations.
    #[must_use]
    pub fn with_seed(mut self, shuffle_seed: u64) -> Self {
        self.shuffle_seed = shuffle_seed;
        self
    }

    /// Reset the internal epoch counter so the next shuffled iterator reproduces epoch zero.
    pub fn reset(&self) {
        self.epoch.set(0);
    }

    /// Number of samples in the dataset.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.features.dims()[0]
    }

    /// Number of feature columns.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.features.dims()[1]
    }

    /// Number of target columns.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.dims()[1]
    }

    /// Batch sizing mode used by this loader.
    #[must_use]
    pub fn batch_size(&self) -> BatchSize {
        self.batch_size
    }

    /// Number of batches produced by one full iterator pass.
    #[must_use]
    pub fn num_batches(&self) -> usize {
        self.sample_count().div_ceil(self.resolved_batch_size())
    }

    /// Create an iterator for the next epoch.
    ///
    /// When shuffling is enabled, each call advances the epoch counter and produces a
    /// different deterministic permutation derived from the configured seed.
    #[must_use]
    pub fn iter(&self) -> DataLoaderIter<'_> {
        if !self.shuffle {
            return self.iter_with_epoch(0);
        }

        let epoch = self.epoch.get();
        self.epoch.set(epoch + 1);
        self.iter_with_epoch(epoch)
    }

    /// Create an iterator for a specific deterministic epoch index.
    #[must_use]
    pub fn iter_with_epoch(&self, epoch: u64) -> DataLoaderIter<'_> {
        // Skip permutation work when shuffle would produce nothing
        // observable: at most one batch is order-invariant (the batch
        // IS the dataset, and the model's loss doesn't depend on the
        // order of rows within a batch). Elides a `sample_count * 4`
        // byte `index_select` on `BatchSize::All` callers that set
        // `with_shuffle(true)` without meaning to.
        let effective_shuffle = self.shuffle && self.num_batches() > 1;
        let shuffled_indices = if effective_shuffle {
            let mut scratch = self.shuffle_indices_scratch.borrow_mut();
            build_shuffled_indices(&mut scratch, self.sample_count(), self.shuffle_seed, epoch);
            Some(std::mem::take(&mut *scratch))
        } else {
            None
        };

        DataLoaderIter {
            loader: self,
            position: 0,
            batch_size: self.resolved_batch_size(),
            shuffle_scratch: &self.shuffle_indices_scratch,
            shuffled_indices,
        }
    }

    fn resolved_batch_size(&self) -> usize {
        self.batch_size.resolve(self.sample_count())
    }

    fn batch_from_range(
        &self,
        start: usize,
        end: usize,
        shuffled_indices: Option<&[u32]>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_len = end - start;
        match shuffled_indices {
            Some(indices) => {
                let batch_indices =
                    Tensor::from_slice(&indices[start..end], batch_len, self.features.device())?;
                Ok((
                    self.features.index_select(&batch_indices, 0)?,
                    self.targets.index_select(&batch_indices, 0)?,
                ))
            }
            None => Ok((
                self.features.narrow(0, start, batch_len)?,
                self.targets.narrow(0, start, batch_len)?,
            )),
        }
    }
}

/// Iterator over [`DataLoader`] batches.
pub struct DataLoaderIter<'loader> {
    loader: &'loader DataLoader,
    position: usize,
    batch_size: usize,
    /// Borrowed unconditionally — if `shuffled_indices` is `None`
    /// (unshuffled path or all-batch short-circuit) the recycle path
    /// in `Drop` bails before ever reaching into the `RefCell`.
    shuffle_scratch: &'loader RefCell<Vec<u32>>,
    shuffled_indices: Option<Vec<u32>>,
}

impl Drop for DataLoaderIter<'_> {
    fn drop(&mut self) {
        let Some(indices) = self.shuffled_indices.take() else {
            return;
        };
        let mut scratch = self.shuffle_scratch.borrow_mut();
        if scratch.is_empty() {
            *scratch = indices;
        }
    }
}

impl DataLoaderIter<'_> {
    fn remaining_batches(&self) -> usize {
        if self.position >= self.loader.sample_count() {
            0
        } else {
            (self.loader.sample_count() - self.position).div_ceil(self.batch_size)
        }
    }
}

impl Iterator for DataLoaderIter<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.loader.sample_count() {
            return None;
        }

        let start = self.position;
        let end = (self.position + self.batch_size).min(self.loader.sample_count());
        self.position = end;

        Some(
            self.loader
                .batch_from_range(start, end, self.shuffled_indices.as_deref()),
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining_batches();
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for DataLoaderIter<'_> {
    fn len(&self) -> usize {
        self.remaining_batches()
    }
}

impl FusedIterator for DataLoaderIter<'_> {}

impl<'loader> IntoIterator for &'loader DataLoader {
    type Item = Result<(Tensor, Tensor)>;
    type IntoIter = DataLoaderIter<'loader>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl sparam_core::batch_source::BatchSource for DataLoader {
    fn batches(&self) -> Box<dyn Iterator<Item = Result<(Tensor, Tensor)>> + '_> {
        Box::new(self.iter())
    }

    fn sample_count(&self) -> usize {
        self.sample_count()
    }

    fn num_batches(&self) -> usize {
        self.num_batches()
    }
}

fn build_shuffled_indices(indices: &mut Vec<u32>, sample_count: usize, seed: u64, epoch: u64) {
    // The `as u32` cast below is only sound because
    // `validate_dataset_pair` rejects `sample_count > u32::MAX` at
    // `DataLoader` construction time — this function is private and
    // only reachable from that path. The debug_assert catches future
    // callers that skip the validator (tests, direct harness use).
    debug_assert!(
        sample_count <= u32::MAX as usize,
        "build_shuffled_indices called with sample_count={sample_count} > u32::MAX; \
         DataLoader construction must reject such sizes"
    );
    let epoch_seed = seed.wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut rng = ChaCha8Rng::seed_from_u64(epoch_seed);
    indices.clear();
    indices.extend((0..sample_count).map(|index| index as u32));
    indices.shuffle(&mut rng);
}

fn validate_dataset_pair(features: &Tensor, targets: &Tensor) -> Result<()> {
    let (feature_samples, _) = validate_matrix(features, "DataLoader::new", "features")?;
    let (target_samples, _) = validate_matrix(targets, "DataLoader::new", "targets")?;

    if feature_samples != target_samples {
        return Err(candle_msg(format!(
            "DataLoader::new expected features and targets to have the same number of samples, got {feature_samples} and {target_samples}"
        )));
    }
    if feature_samples > u32::MAX as usize {
        return Err(candle_msg(format!(
            "DataLoader::new currently supports at most {} samples for indexed batching, got {feature_samples}",
            u32::MAX
        )));
    }
    if !features.device().same_device(targets.device()) {
        return Err(candle_msg(
            "DataLoader::new expects features and targets to live on the same device",
        ));
    }

    Ok(())
}

fn validate_matrix(tensor: &Tensor, context: &str, name: &str) -> Result<(usize, usize)> {
    let dims = tensor.dims();
    if dims.len() != 2 {
        return Err(candle_msg(format!(
            "{context} expects {name} to be a 2D tensor shaped (samples, features), got {:?}",
            dims
        )));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(candle_msg(format!(
            "{context} expects {name} to have non-empty sample and feature dimensions, got {:?}",
            dims
        )));
    }
    Ok((dims[0], dims[1]))
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;
    use crate::scaling::{MinMaxScaler, Scaler, StandardScaler};

    fn collect_feature_order(iter: DataLoaderIter<'_>) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        for batch in iter {
            let (features, _) = batch?;
            values.extend(features.flatten_all()?.to_vec1::<f64>()?);
        }
        Ok(values)
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_dataloader_yields_fixed_batches_and_last_remainder() -> Result<()> {
        let features = Tensor::from_vec(
            vec![1.0f64, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
            (5, 2),
            &Device::Cpu,
        )?;
        let targets = Tensor::from_vec(
            vec![101.0f64, 102.0, 103.0, 104.0, 105.0],
            (5, 1),
            &Device::Cpu,
        )?;
        let loader = DataLoader::new(features, targets, BatchSize::Fixed(2))?;

        let batches = loader.iter().collect::<Result<Vec<_>>>()?;

        assert_eq!(loader.sample_count(), 5);
        assert_eq!(loader.num_batches(), 3);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.dims(), &[2, 2]);
        assert_eq!(batches[1].0.dims(), &[2, 2]);
        assert_eq!(batches[2].0.dims(), &[1, 2]);
        assert_eq!(batches[2].1.dims(), &[1, 1]);

        let first_features = batches[0].0.flatten_all()?.to_vec1::<f64>()?;
        let last_targets = batches[2].1.flatten_all()?.to_vec1::<f64>()?;
        assert_eq!(first_features, vec![1.0, 10.0, 2.0, 20.0]);
        assert_eq!(last_targets, vec![105.0]);

        Ok(())
    }

    #[test]
    fn test_dataloader_iterator_reports_exact_remaining_batch_count() -> Result<()> {
        let features =
            Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &Device::Cpu)?;
        let targets = Tensor::from_vec(vec![10.0f64, 20.0, 30.0], (3, 1), &Device::Cpu)?;
        let loader = DataLoader::new(features, targets, BatchSize::Fixed(2))?;
        let mut iter = loader.iter();

        assert_eq!(iter.len(), 2);
        assert_eq!(iter.size_hint(), (2, Some(2)));

        let _first = iter.next().expect("first batch should exist")?;
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.size_hint(), (1, Some(1)));

        let _second = iter.next().expect("second batch should exist")?;
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.size_hint(), (0, Some(0)));
        assert!(iter.next().is_none());

        Ok(())
    }

    #[test]
    fn test_dataloader_supports_all_batch_size_and_dataset_column_split() -> Result<()> {
        let dataset = Tensor::from_vec(
            vec![
                1.0f64, 2.0, 3.0, 4.0, 10.0, 20.0, 5.0, 6.0, 7.0, 8.0, 30.0, 40.0,
            ],
            (2, 6),
            &Device::Cpu,
        )?;
        let loader = DataLoader::from_dataset_columns(dataset, 4, BatchSize::All)?;
        let mut iter = loader.iter();

        let (features, targets) = iter.next().expect("one batch should be yielded")?;

        assert_eq!(loader.num_batches(), 1);
        assert_eq!(loader.feature_count(), 4);
        assert_eq!(loader.target_count(), 2);
        assert_eq!(features.dims(), &[2, 4]);
        assert_eq!(targets.dims(), &[2, 2]);
        assert!(iter.next().is_none());

        Ok(())
    }

    /// Splits via [`DataLoader::from_dataset_columns`] are `narrow`
    /// views, not materialised tensors. Verify they compose correctly
    /// with `index_select`-based shuffled batching (which is the
    /// path used when `BatchSize::Fixed(n)` + `with_shuffle(true)`).
    #[test]
    fn test_dataloader_from_dataset_columns_with_shuffle_preserves_row_pairing() -> Result<()> {
        // 4 rows × 3 cols; split as feature_count=2 → features[col0,col1], targets[col2].
        // Row values are keyed by row index so a permutation is easy to verify.
        let dataset = Tensor::from_vec(
            vec![
                1.0f64, 10.0, 100.0,
                2.0,    20.0, 200.0,
                3.0,    30.0, 300.0,
                4.0,    40.0, 400.0,
            ],
            (4, 3),
            &Device::Cpu,
        )?;
        let loader = DataLoader::from_dataset_columns(dataset, 2, BatchSize::Fixed(2))?
            .with_shuffle(true)
            .with_seed(123);

        let mut feature_rows = Vec::new();
        let mut target_rows = Vec::new();
        for batch in loader.iter_with_epoch(0) {
            let (features, targets) = batch?;
            feature_rows.extend(features.to_vec2::<f64>()?);
            target_rows.extend(targets.flatten_all()?.to_vec1::<f64>()?);
        }

        // Every row must still pair its target: target == col0 * 100.
        // (Guards against a future regression where `narrow` views +
        // `index_select` drift into selecting features and targets
        // with different permutations.)
        assert_eq!(feature_rows.len(), 4);
        assert_eq!(target_rows.len(), 4);
        for (feat, &target) in feature_rows.iter().zip(target_rows.iter()) {
            let col0 = feat[0];
            assert!(
                (target - col0 * 100.0).abs() < 1e-12,
                "row pairing broken: col0={col0}, target={target}",
            );
        }

        // Shuffled order must differ from the identity — otherwise the
        // test degenerates into the unshuffled case.
        let identity: Vec<f64> = (1..=4).map(|i| i as f64).collect();
        let shuffled_col0: Vec<f64> = feature_rows.iter().map(|r| r[0]).collect();
        assert_ne!(shuffled_col0, identity);

        Ok(())
    }

    #[test]
    fn test_dataloader_shuffle_is_seeded_epoch_specific_and_resettable() -> Result<()> {
        let features =
            Tensor::from_vec(vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0], (6, 1), &Device::Cpu)?;
        let targets = Tensor::from_vec(
            vec![10.0f64, 11.0, 12.0, 13.0, 14.0, 15.0],
            (6, 1),
            &Device::Cpu,
        )?;

        let loader_a = DataLoader::new(features.clone(), targets.clone(), BatchSize::Fixed(2))?
            .with_shuffle(true)
            .with_seed(7);
        let loader_b = DataLoader::new(features, targets, BatchSize::Fixed(2))?
            .with_shuffle(true)
            .with_seed(7);

        let epoch_zero_a = collect_feature_order(loader_a.iter_with_epoch(0))?;
        let epoch_zero_b = collect_feature_order(loader_b.iter_with_epoch(0))?;
        let epoch_one_a = collect_feature_order(loader_a.iter_with_epoch(1))?;

        assert_eq!(epoch_zero_a, epoch_zero_b);
        assert_ne!(epoch_zero_a, epoch_one_a);

        let first_epoch = collect_feature_order(loader_a.iter())?;
        let second_epoch = collect_feature_order(loader_a.iter())?;
        assert_ne!(first_epoch, second_epoch);

        loader_a.reset();
        let reset_epoch = collect_feature_order(loader_a.iter())?;
        assert_eq!(first_epoch, reset_epoch);

        Ok(())
    }

    #[test]
    fn test_dataloader_returns_prescaled_tensors_unchanged() -> Result<()> {
        // The loader no longer transforms per batch — callers feed
        // already-scaled tensors. Verify that feeding a StandardScaler
        // output in produces bit-identical output out, and that
        // MinMaxScaler-normalised targets come through unchanged.
        let features = Tensor::from_vec(
            vec![1.0f64, 10.0, 3.0, 10.0, 5.0, 10.0],
            (3, 2),
            &Device::Cpu,
        )?;
        let targets = Tensor::from_vec(vec![100.0f64, 200.0, 300.0], (3, 1), &Device::Cpu)?;
        let mut feature_scaler = StandardScaler::new();
        let mut target_scaler = MinMaxScaler::new();

        feature_scaler.fit(&features)?;
        target_scaler.fit(&targets)?;

        // Pre-scale once — this is what happens at the call site now.
        let scaled_features_src = feature_scaler.transform(&features)?;
        let scaled_targets_src = target_scaler.transform(&targets)?;

        let loader = DataLoader::new(
            scaled_features_src,
            scaled_targets_src,
            BatchSize::All,
        )?;
        let (scaled_features, scaled_targets) =
            loader.iter().next().expect("one batch should be yielded")?;

        let scaled_features = scaled_features.flatten_all()?.to_vec1::<f64>()?;
        let scaled_targets = scaled_targets.flatten_all()?.to_vec1::<f64>()?;

        assert_close(scaled_features[0], -1.224_744_871_391_589);
        assert_close(scaled_features[1], 0.0);
        assert_close(scaled_features[2], 0.0);
        assert_close(scaled_features[3], 0.0);
        assert_close(scaled_features[4], 1.224_744_871_391_589);
        assert_close(scaled_features[5], 0.0);
        assert_close(scaled_targets[0], 0.0);
        assert_close(scaled_targets[1], 0.5);
        assert_close(scaled_targets[2], 1.0);

        Ok(())
    }

    #[test]
    fn test_dataloader_rejects_invalid_configuration() -> Result<()> {
        let features = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let mismatched_targets = Tensor::from_vec(vec![1.0f64, 2.0, 3.0], (3, 1), &Device::Cpu)?;
        let dataset = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu)?;

        let mismatch_error =
            DataLoader::new(features.clone(), mismatched_targets, BatchSize::Fixed(2))
                .err()
                .expect("mismatched sample counts should be rejected");
        let batch_size_error = DataLoader::new(
            features,
            Tensor::from_vec(vec![1.0f64, 2.0], (2, 1), &Device::Cpu)?,
            BatchSize::Fixed(0),
        )
        .err()
        .expect("zero batch size should be rejected");
        let split_error = DataLoader::from_dataset_columns(dataset, 4, BatchSize::All)
            .err()
            .expect("feature_count equal to total column count should be rejected");

        assert!(
            mismatch_error
                .to_string()
                .contains("same number of samples")
        );
        assert!(batch_size_error.to_string().contains("greater than zero"));
        assert!(split_error.to_string().contains("feature_count"));

        Ok(())
    }

    #[test]
    fn test_dataloader_large_fixed_batch_size_returns_single_batch() -> Result<()> {
        let features = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        let targets = Tensor::from_vec(vec![10.0f64, 20.0], (2, 1), &Device::Cpu)?;
        let loader = DataLoader::new(features, targets, BatchSize::Fixed(8))?;

        let batches = loader.iter().collect::<Result<Vec<_>>>()?;

        assert_eq!(loader.num_batches(), 1);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0.dims(), &[2, 2]);

        Ok(())
    }
}
