import datetime as dt
import os

import polars as pl
import pyarrow
import pyarrow.parquet

from .dataset import Dataset, lex_min, lex_max
from .ordering import lex_key, columns_geq, columns_lt
from .metadata import PARTITION_NUMBER_DIGITS
from .progress import make_progress_callback


def partition_index_expr(index_columns, divisions):
    if len(divisions) <= 2:
        return pl.lit(0)
    for i, (lb, ub) in enumerate(zip(divisions[:-1], divisions[1:])):
        cond = pl.lit(True)
        if lb is None:
            expr = pl.when(columns_lt(index_columns, ub)).then(i)
        elif ub is None:
            expr = expr.otherwise(i)
        else:
            expr = expr.when(columns_lt(index_columns, ub)).then(i)
    return expr


def get_row_divisions(partition_sizes, rows_per_partition):
    num_rows = sum(partition_sizes)
    df_old = (
        pl.DataFrame(
            pl.Series('row', [0] + list(partition_sizes)[:-1], pl.UInt32))
        .with_row_index('part_index')
        .with_columns(
            row=pl.col('row').cum_sum(),
            is_new_div=pl.lit(False, pl.Boolean),
        )
        .select(
            'row',
            'part_index',
            pl.col('row').alias('part_base'),
            'is_new_div',
        )
    )
    df_new = (
        pl.DataFrame(
            pl.int_range(
                0, num_rows, rows_per_partition, dtype=pl.UInt32, eager=True)
            .alias('row')
        )
        .with_columns(
            part_index=pl.lit(None, pl.UInt32),
            part_base=pl.lit(None, pl.UInt32),
            is_new_div=pl.lit(True, pl.Boolean),
        )
    )
    divisions = list(
        pl.concat([df_old, df_new])
        .sort('row', 'is_new_div')
        .with_columns(
            part_index=pl.col('part_index').fill_null(strategy='forward'),
            part_base=pl.col('part_base').fill_null(strategy='forward'),
        )
        .with_columns(offset=pl.col('row') - pl.col('part_base'))
        .filter(pl.col('is_new_div') & (pl.col('row') > 0))
        .select('part_index', 'offset')
        .rows()
    )

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
        .group_by(index_columns)
        .agg(pl.len().alias('__size'))
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
        .group_by(index_columns)
        .agg(pl.col('__size').sum())
        .sort(index_columns)
        .with_columns(__part=pl.col('__size').cum_sum()//samples_per_partition)
        .collect()
    )
    lower_bounds = list(
        sample
        .group_by('__part')
        .head(1)
        .sort('__part')
        .select(index_columns)
        .rows()
    )
    divisions = lower_bounds[1:]
    if samples_per_partition == rows_per_partition:
        upper_bounds = list(
            sample
            .group_by('__part')
            .tail(1)
            .sort('__part')
            .select(index_columns)
            .rows()
        )
        sizes = (
            sample
            .group_by('__part')
            .agg(pl.col('__size').sum())
            .sort('__part')
            .get_column('__size')
            .to_list()
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
            index_columns=None,
            sample_fraction=1.0,
            parallel=False,
            progress=False,
            base_seed=10,
            seed_increment=10,
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        self._other = other

        if index_columns is None:
            index_columns = self._other.index_columns
        if not index_columns:
            if not self._other.known_sizes:
                self._other = self._other.reindex()
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
        self._cached_partition_index = None
        self._cached_partition = None

        super().__init__(
            npartitions=len(divisions) + 1,
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=self._other._schema,
        )
        self._divisions = [None] + divisions + [None]

    def _get_partition(self, partition_index, cache=False):
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
                if cache:
                    if self._cached_partition_index is None \
                            or self._cached_partition_index != i_part:
                        self._cached_partition = self._other[i_part].collect()
                        self._cached_partition_index = i_part
                    part = self._cached_partition.lazy()
                else:
                    part = self._other[i_part]
                row_beg = from_row if i_part == from_part else None
                row_end = to_row if i_part == to_part else None
                if row_beg is None and row_end is None:
                    parts.append(part)
                elif row_beg is None:
                    parts.append(part.slice(0, row_end))
                elif row_end is None:
                    parts.append(part.slice(row_beg))
                else:
                    parts.append(part.slice(row_beg, row_end-row_beg))
            return pl.concat(parts).lazy()

    def _fast_write_parquet(self, path, progress):
        progress = make_progress_callback(progress)
        index_columns = self._index_columns
        filenames = [
            f'part{{0:0>{PARTITION_NUMBER_DIGITS}d}}.parquet'.format(i)
            for i in range(self._npartitions)
        ]
        lower_bounds = [None]*self._npartitions
        upper_bounds = [None]*self._npartitions
        sizes = [0]*self._npartitions
        schema = self._other.schema
        self._init_path(path, append=False, default_schema=schema)
        t0 = dt.datetime.now()
        if index_columns:
            try:
                writers = [None]*self._npartitions
                for i_part, part in enumerate(self._other):
                    part = (
                        part
                        .collect()
                        .with_columns(
                            __part=partition_index_expr(
                                index_columns,
                                self._divisions,
                            )
                        )
                    )
                    for (i,), batch in \
                            part.partition_by('__part', as_dict=True).items():
                        batch = batch.drop('__part')
                        batch_lb = lex_min(batch.select(index_columns))
                        batch_ub = lex_max(batch.select(index_columns))
                        if lower_bounds[i] is None:
                            lower_bounds[i] = batch_lb
                        else:
                            lower_bounds[i] = min(
                                lower_bounds[i], batch_lb, key=lex_key)
                        if upper_bounds[i] is None:
                            upper_bounds[i] = batch_ub
                        else:
                            upper_bounds[i] = max(
                                upper_bounds[i], batch_ub, key=lex_key)
                        sizes[i] += len(batch)
                        if schema is None:
                            schema = batch.schema
                        batch = batch.to_arrow()
                        if writers[i] is None:
                            writers[i] = pyarrow.parquet.ParquetWriter(
                                os.path.join(path, filenames[i]),
                                batch.schema,
                                compression='ZSTD',
                            )
                        writers[i].write_table(batch)
                    progress(
                        i_part + 1,
                        self._other._npartitions,
                        dt.datetime.now(),
                        t0,
                    )
            finally:
                for writer in writers:
                    if writer is not None:
                        writer.close()
        else:
            try:
                for i, (lb, ub) in enumerate(zip(
                        self._divisions[:-1], self._divisions[1:])):
                    part = self._get_partition(i, cache=True).collect()
                    lower_bounds[i] = ()
                    upper_bounds[i] = ()
                    sizes[i] = len(part)
                    if schema is None:
                        schema = part.schema
                    if len(part) > 0:
                        part.write_parquet(os.path.join(path, filenames[i]))
            finally:
                self._cached_partition_index = None
                self._cached_partition = None

        filenames = [f for f, s in zip(filenames, sizes) if s > 0]
        lower_bounds = [lb for lb, s in zip(lower_bounds, sizes) if s > 0]
        upper_bounds = [lb for lb, s in zip(upper_bounds, sizes) if s > 0]
        max_partition_index = max(i for i, s in enumerate(sizes) if s > 0)
        sizes = [s for s in sizes if s > 0]
        self._write_metadata(
            path,
            (
                filenames,
                sizes,
                lower_bounds,
                upper_bounds,
                schema,
            ),
            max_partition_index,
        )
        return self._read_persisted(path)

    def write_parquet(
            self,
            path,
            append=False,
            parallel=False,
            progress=False,
    ):
        if not append and not parallel:
            return self._fast_write_parquet(path, progress=progress)
        return super().write_parquet(
            path, append=append, parallel=parallel, progress=progress)


def _repartition(
    self,
    rows_per_partition,
    index_columns=None,
    sample_fraction=1.0,
    parallel=False,
    progress=False,
    base_seed=10,
    seed_increment=10,
):
    """Repartition the dataset.

    The data is partitioned so that rows with the same values for
    the index columns appear in the same partition.

    Note:
      On large datasets repartitioning is an expensive operation. It is
      generally recommended to persist the dataset immediately after
      repartitioning by calling :py:meth:`padawan.Dataset.write_parquet`.
      There is an optimised implementation of ``write_parquet`` for
      repartitioned datasets. To use this implementation call ``write_parquet``
      directly after ``repartition`` and make sure that the `parallel` and
      `append` arguments are set to their default values (i.e. ``False``).

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
    if len(self) == 0:
        return self
    return RepartitionedDataset(
        self,
        rows_per_partition,
        index_columns=index_columns,
        sample_fraction=sample_fraction,
        parallel=parallel,
        progress=progress,
        base_seed=base_seed,
        seed_increment=seed_increment,
    )

Dataset.repartition = _repartition

