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
        sample = sample.collect().sample(frac=frac, seed=seed).lazy()
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
        .collect(parallel=parallel)
        .lazy()
        .groupby(index_columns)
        .agg(pl.col('__size').sum())
        .sort(index_columns)
        .with_column(
            (pl.col('__size').cumsum()/samples_per_partition)
            .ceil().cast(pl.Int32)
            .alias('__part')
        )
        .collect()
    )
    lower_bounds = list(
        sample
        .groupby('__part')
        .first()
        .sort('__part')
        .select(index_columns)
        .rows()
    )
    divisions = lower_bounds[1:]

    if samples_per_partition == rows_per_partition:
        upper_bounds = list(
            sample
            .groupby('__part')
            .last()
            .sort('__part')
            .select(index_columns)
            .rows()
        )
        sizes = list(
            sample
            .groupby('__part')
            .agg(pl.col('__size').sum())
            .sort('__part')
            .get_column('__size')
        )
    else:
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
                .slice(lb, ub)
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
        base_seed=10,
        seed_increment=10,
):
    return RepartitionedDataset(
        self,
        rows_per_partition,
        sample_fraction=sample_fraction,
        parallel=parallel,
        base_seed=base_seed,
        seed_increment=seed_increment,
    )

Dataset.repartition = _repartition

