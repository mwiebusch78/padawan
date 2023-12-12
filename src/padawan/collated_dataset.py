import polars as pl

from .dataset import Dataset
from .ordering import lex_key, sort_partitions


class CollatedDataset(Dataset):
    def __init__(self, other, rows_per_partition):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        if not other.known_sizes and other.known_bounds:
            raise ValueError(
                'Stats must be known to use collate. Use reindex first.')
        self._other = other

        other_lower_bounds = self._other.lower_bounds
        other_upper_bounds = self._other.upper_bounds
        other_sizes = self._other.sizes
        partition_indices = sort_partitions(
            other_lower_bounds, other_upper_bounds)

        batches = []
        lower_bounds = []
        upper_bounds = []
        sizes = []

        batch = []
        size = 0
        lb = None
        ub = None

        def add_batch():
            nonlocal batch, size, lb, ub
            batches.append(batch)
            sizes.append(size)
            lower_bounds.append(lb)
            upper_bounds.append(ub)
            batch = []
            size = 0
            lb = None
            ub = None

        for i in partition_indices:
            batch.append(i)
            size += other_sizes[i]
            if lb is None:
                lb = other_lower_bounds[i]
                ub = other_upper_bounds[i]
            else:
                lb = min(lb, other_lower_bounds[i], key=lex_key)
                ub = max(ub, other_upper_bounds[i], key=lex_key)
            if size >= rows_per_partition:
                add_batch()
        if batch:
            add_batch()
            
        super().__init__(
            npartitions=len(batches),
            index_columns=self._other.index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=self._other._schema,
        )
        self._batches = batches

    def _get_partition(self, partition_index):
        batch = self._batches[partition_index]
        parts = [self._other[i] for i in batch]
        return pl.concat(parts)


def _collate(self, rows_per_partition):
    """Merge partitions to get a certain minimum number of rows per partition.

    The partition sizes and bounds must be known to use this method. You can
    call :py:meth:`padawan.Dataset.reindex` first to compute them.

    This method does not split existing partitions. Use
    :py:meth:`padawan.Dataset.repartition` for better (but computationally
    more expensive) control over partition sizes.

    Args:
      rows_per_partition (int): The desired minimum number of rows per
        partition.

    Returns:
      padawan.Dataset: A dataset with the desired minimum number of
        rows per partition.
    """
    return CollatedDataset(self, rows_per_partition)
Dataset.collate = _collate
