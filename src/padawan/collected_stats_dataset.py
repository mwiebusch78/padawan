import polars as pl

from .dataset import Dataset
from .parallelize import parallel_map


class CollectedStatsDataset(Dataset):
    def _get_partition_stats(self, partition_index):
        _, nrows, lb, ub \
            = self._other._get_partition_with_stats(partition_index)
        return nrows, lb, ub

    def __init__(self, other, parallel=False):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        self._other = other

        partition_indices = list(range(len(self._other)))
        stats = parallel_map(
            self._get_partition_stats,
            partition_indices,
            workers=parallel,
        )
        sizes = [s[0] for s in stats]
        lower_bounds = [s[1] for s in stats]
        upper_bounds = [s[2] for s in stats]

        partition_indices = [
            i for i, s in zip(partition_indices, sizes) if s > 0]
        if len(partition_indices) < len(self._other):
            lower_bounds = [lower_bounds[i] for i in partition_indices]
            upper_bounds = [upper_bounds[i] for i in partition_indices]
            sizes = [sizes[i] for i in partition_indices]
            partition_indices = tuple(partition_indices)
        else:
            partition_indices = None

        super().__init__(
            npartitions=len(sizes),
            index_columns=self._other.index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self._partition_indices = partition_indices

    def __getitem__(self, partition_index):
        if self._partition_indices is None:
            return self._other[partition_index]
        return self._other[self._partition_indices[partition_index]]


def _collect_stats(self, parallel=False):
    """Compute partition sizes and bounds if they are not known.

    The `sizes`, `lower_bounds` and `upper_bounds` properties will be
    set after calling `_compute_stats`.

    Args:
      parallel (bool or int): Specifies how to parallelize computation:
        `parallel = True` -- use all available CPUs
        `parallel = False` -- no parallelism
        `parallel > 1` -- use `parallel` number of CPUs
        `parallel in [0, 1]` -- no parallelism
        `parallel = -n < 0` -- use number of available CPUs minus n
    """
    return CollectedStatsDataset(self, parallel=parallel)

Dataset.collect_stats = _collect_stats

