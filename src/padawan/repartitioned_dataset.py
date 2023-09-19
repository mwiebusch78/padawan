import polars as pl

from .dataset import Dataset


def get_row_divisions(partition_sizes, rows_per_partition):
        partition_indices = []
        row_indices = []
        src_partition = 0
        src_row_index = 0
        dest_row_index = 0
        num_src_partitions = len(partition_sizes)
        while (src_partition, src_row_index) < (num_src_partitions, 0):
            rem_src_rows = partition_sizes[src_partition] - src_row_index
            rem_dest_rows = rows_per_partition - dest_row_index
            if rem_src_rows < rem_dest_rows:
                # add remainder of source partition
                dest_row_index += rem_src_rows
                src_partition += 1
                src_row_index = 0
            elif rem_src_rows == rem_dest_rows:
                # add remainder of source partition and create split
                dest_row_index += rem_src_rows
                src_partition += 1
                src_row_index = 0
                dest_row_index = 0
                if src_partition < num_src_partitions:
                    partition_indices.append(src_partition)
                    row_indices.append(src_row_index)
            else:
                # add part of source partition and create split
                src_row_index += rem_dest_rows
                dest_row_index = 0
                if src_partition < num_src_partitions:
                    partition_indices.append(src_partition)
                    row_indices.append(src_row_index)

        divisions = list(zip(partition_indices, row_indices))
        sizes = [rows_per_partition]*len(divisions) + \
            [sum(partition_sizes)-rows_per_partition*len(divisions)]
        lower_bounds = [()]*len(sizes)
        upper_bounds = [()]*len(sizes)
        return divisions, sizes, lower_bounds, upper_bounds


def _sample_partition(part, seed, index_columns, frac):
    sample = part.select(index_columns)
    if frac < 1.0:
        sample = sample.collect()
        nrows = max(int(frac*len(sample)), 1)
        sample = sample.sample(nrows, seed=seed).lazy()
    sample = (
        sample
        .groupby(index_columns)
        .agg(pl.count().alias('__size'))
    )
    return sample


def get_index_divisions(
        ds,
        rows_per_partition,
        sample_fraction,
        index_columns,
        base_seed,
        seed_increment,
        parallel,
        progress,
):
    sample_fraction = min(sample_fraction, 1.0)
    samples_per_partition = max(1, int(sample_fraction*rows_per_partition))

    extra_args = [
        (base_seed + i*seed_increment,) for i in range(len(ds))]
    shared_args = (index_columns, sample_fraction)
    sample = (
        ds
        .map(
            _sample_partition,
            extra_args=extra_args,
            shared_args=shared_args,
        )
        .collect(parallel=parallel, progress=progress)
        .lazy()
        .groupby(index_columns)
        .agg(pl.col('__size').sum())
        .sort(index_columns)
        .with_columns(pl.col('__size').cumsum())
        .collect()
    )

    lower_bounds = []
    upper_bounds = []
    sizes = []
    current_size = 0
    while len(sample) > 0:
        head = sample.filter(
            pl.col('__size') <= current_size + samples_per_partition)
        if len(head) == 0:
            head = sample[:1, :]
            sample = sample[1:, :]
        else:
            sample = sample.filter(
                pl.col('__size') > current_size + samples_per_partition)
        lower_bounds.append(head.row(0)[:-1])
        upper_bounds.append(head.row(-1)[:-1])
        sizes.append(head[-1, '__size'] - current_size)
        current_size = head[-1, '__size']

    divisions = lower_bounds[1:]

    if samples_per_partition != rows_per_partition:
        lower_bounds = None
        upper_bounds = None
        sizes = None

    return divisions, sizes, lower_bounds, upper_bounds


class RepartitionedDataset(Dataset):
    def __init__(
            self,
            other,
            rows_per_partition,
            sample_fraction=1.0,
            parallel=False,
            progress=False,
            base_seed=10,
            seed_increment=10,
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        if not other.known_sizes and other.known_bounds:
            raise ValueError(
                'Stats must be known when using repartition. Try calling '
                'reindex() first.')
        self._other = other

        index_columns = self._other.index_columns
        if not index_columns:
            divisions, sizes, lower_bounds, upper_bounds \
                = get_row_divisions(self._other.sizes, rows_per_partition)
        else:
            divisions, sizes, lower_bounds, upper_bounds \
                = get_index_divisions(
                    ds=self._other,
                    rows_per_partition=rows_per_partition,
                    sample_fraction=sample_fraction,
                    index_columns=index_columns,
                    base_seed=base_seed,
                    seed_increment=seed_increment,
                    parallel=parallel,
                    progress=progress,
                )

        super().__init__(
            npartitions=len(divisions) + 1,
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=self._other._schema,
        )
        self._divisions = [None] + divisions + [None]

    def _get_partition(self, partition_index):
        lb = self._divisions[partition_index]
        ub = self._divisions[partition_index + 1]
        if self.index_columns:
            return (
                self._other
                .slice(lb, ub, inclusive='lower')
                .collect()
                .lazy()
            )
        else:
            from_part = 0
            to_part = len(self._other) - 1
            from_row = None
            to_row = None
            if lb is not None:
                from_part, from_row = lb
            if ub is not None:
                to_part, to_row = ub

            parts = []
            for i_part in range(from_part, to_part + 1):
                part = self._other[i_part].collect()
                row_beg = from_row if i_part == from_part else None
                row_end = to_row if i_part == to_part else None
                parts.append(part[slice(row_beg, row_end), :])
            return pl.concat(parts).lazy()


def _repartition(
        self, 
        rows_per_partition,
        sample_fraction=1.0,
        parallel=False,
        progress=False,
        base_seed=10,
        seed_increment=10,
):
    """Repartition the dataset.

    The data is partitioned so that rows with the same values for
    the index columns appear in the same partition.

    Args:
      rows_per_partition (int): The desired number of rows per partition.
      sample_fraction (float, optional): The fraction of rows of the full
        dataset that will be sampled in order to determine the new partition
        boundaries. Defaults to 1, in which case all rows are processed.
        You need to reduce this in cases where the index columns for the full
        dataset cannot be stored in memory.
      parallel (bool or int, optional): Specifies how to parallelize the
        computation. See corresponding argument for
        :py:meth:`padawan.Dataset.write_parquet` for details.
        Defaults to ``False``.
      progress (callable, str, int, bool or tuple, optional):
        Whether and how to print progress messages. See corresponding
        argument for :py:meth:`padawan.Dataset.write_parquet` for details.
        Defaults to ``False``.
      base_seed (int, optional): The random seed used to sample rows from
        the first partition of `self`. Defaults to 10.
      seed_increment (int, optional): For every subsequent partition the
        random seed in incremented by `seed_increment`. Defaults to 10.

    Returns:
      padawan.Dataset: The repartitioned dataset.

    """
    return RepartitionedDataset(
        self,
        rows_per_partition,
        sample_fraction=sample_fraction,
        parallel=parallel,
        progress=progress,
        base_seed=base_seed,
        seed_increment=seed_increment,
    )

Dataset.repartition = _repartition

