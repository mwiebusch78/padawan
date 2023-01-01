import polars as pl

from .dataset import Dataset
from .parallelize import parallel_map


class ReindexedDataset(Dataset):
    def _get_partition_stats(self, partition_index, index_columns):
        _, nrows, lb, ub = self._other._get_partition_with_stats(
                partition_index, index_columns=index_columns)
        return nrows, lb, ub

    def __init__(
            self,
            other,
            index_columns=None,
            collect_stats=True,
            parallel=False,
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        self._other = other

        if index_columns is None:
            index_columns = self._other.index_columns
        else:
            index_columns = tuple(index_columns)

        if self._other.known_bounds and self._other.known_sizes and \
                len(index_columns) <= len(self._other.index_columns) \
                and index_columns == \
                self._other.index_columns[:len(index_columns)]:
            partition_indices = None
            npartitions = len(self._other)
            sizes = self._other.sizes
            lower_bounds = [
                b[:len(index_columns)] for b in self._other.lower_bounds]
            upper_bounds = [
                b[:len(index_columns)] for b in self._other.upper_bounds]
        elif collect_stats:
            partition_indices = list(range(len(self._other)))
            stats = parallel_map(
                self._get_partition_stats,
                partition_indices,
                shared_args={'index_columns': index_columns},
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
            npartitions = len(sizes)
        else:
            partition_indices = None
            npartitions = len(self._other)
            sizes = self._other.sizes
            lower_bounds = None
            upper_bounds = None

        super().__init__(
            npartitions=npartitions,
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self._partition_indices = partition_indices

    def _get_partition(self, partition_index):
        if self._partition_indices is None:
            return self._other[partition_index]
        return self._other[self._partition_indices[partition_index]]


def _reindex(self, index_columns=None, collect_stats=True, parallel=False):
    """Set index columns and compute partition sizes and bounds.

    Args:
      index_columns (tuple of str, optional): The columns to used as index.
        If ``None``, partition sizes and bounds are computed for the current
        index columns.
      collect_stats (bool, optional): Whether to compute the sizes and
        index bounds of the partitions if they are not known. Defaults to
        ``True``.
      parallel (bool or int, optional): Specifies how to parallelize the
        computation:

          ``parallel = True`` -- use all available CPUs
          ``parallel = False`` -- no parallelism
          ``parallel > 1`` -- use `parallel` number of CPUs
          ``parallel in [0, 1]`` -- no parallelism
          ``parallel = -n < 0`` -- use number of available CPUs minus n
    """
    if self.known_bounds and self.known_sizes and (
            index_columns is None
            or tuple(index_columns) == self.index_columns):
        return self
    return ReindexedDataset(
        self,
        index_columns=index_columns,
        collect_stats=collect_stats,
        parallel=parallel,
    )

Dataset.reindex = _reindex

