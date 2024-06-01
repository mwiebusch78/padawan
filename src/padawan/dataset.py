import shutil
import os
import polars as pl

from .parallelize import parallel_map, is_parallel_config
from .json_io import write_json
from .metadata import (
    load_metadata, PARTITION_NUMBER_DIGITS, METADATA_FILE, SCHEMA_FILE)
from .ordering import sort_partitions, lex_key


def lex_min(df):
    if len(df) == 0:
        return None

    columns = list(df.columns)
    for col in columns:
        if df.select(pl.col(col).is_null().any()).row(0)[0]:
            df = df.filter(pl.col(col).is_null())
        else:
            df = df.filter(pl.col(col) == pl.col(col).min())
    return df.row(0)


def lex_max(df):
    if len(df) == 0:
        return None

    columns = list(df.columns)
    for col in columns:
        df = df.filter(pl.col(col) == pl.col(col).max())
    return df.row(0)


def dataframe_from_schema(schema):
    return pl.DataFrame([pl.Series(c, [], dtype=t) for c, t in schema.items()])


def get_partition_stats(part, index_columns):
    nrows = len(part)
    if index_columns:
        index = part.select(index_columns)
        lb = lex_min(index)
        ub = lex_max(index)
    else:
        lb = ()
        ub = ()
    return nrows, lb, ub


class StatsUnknownError(Exception):
    pass


class AppendError(Exception):
    pass


class Dataset:
    """Abstract base class for datasets.

    This class has the semantics of a list of ``polars.LazyFrame`` objects.
    For an instance ``ds`` of this class you can use ``len(ds)`` to get
    the number of partitions, ``ds[i]`` to access the *i*-th partition and
    ``for part in ds`` to iterate over partitions.

    Usually you will use :py:func:`padawan.scan_parquet` to create
    ``Dataset`` instances, so you will not need to instantiate this class
    or any of its derived classes directly. 
    """

    def __init__(
            self,
            npartitions,
            index_columns=(),
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
            schema=None,
    ):
        self._index_columns = tuple(index_columns)

        npartitions = int(npartitions)
        if npartitions < 0:
            raise ValueError('Number of partitions cannot be negative')
        if npartitions == 0 and schema is None:
            raise ValueError(
                'schema must be specified when number of partitions is zero.')
        self._npartitions = npartitions
        self._schema = schema

        self._sizes = None
        if sizes is not None:
            self._sizes = tuple(int(s) for s in sizes)
            if len(self._sizes) != self._npartitions:
                raise ValueError('sizes has the wrong length')

        self._lower_bounds = None
        if lower_bounds is not None:
            self._lower_bounds = tuple(tuple(b) for b in lower_bounds)
            if len(self._lower_bounds) != self._npartitions:
                raise ValueError('lower_bounds has the wrong length')
            if not all(
                    len(b) == len(self._index_columns)
                    for b in self._lower_bounds):
                raise ValueError(
                    'all lower bounds must be tuples with the same length '
                    'as index_columns')

        self._upper_bounds = None
        if upper_bounds is not None:
            self._upper_bounds = tuple(tuple(b) for b in upper_bounds)
            if len(self._upper_bounds) != self._npartitions:
                raise ValueError('upper_bounds has the wrong length')
            if not all(
                    len(b) == len(self._index_columns)
                    for b in self._upper_bounds):
                raise ValueError(
                    'all upper bounds must be tuples with the same length '
                    'as index_columns')

        if not self._index_columns:
            self._lower_bounds = ((),)*self._npartitions
            self._upper_bounds = ((),)*self._npartitions

    @property
    def index_columns(self):
        """A tuple of strings with the index columns of the dataset.

        ``padawan`` tries to keep track of the upper and lower bounds of the
        index columns in each partition. The :py:meth:`padawan.Dataset.slice`
        and :py:meth:`padawan.Dataset.join` methods implement joining and
        slicing on the current index columns.

        """
        return self._index_columns

    @property
    def known_bounds(self):
        """``True`` if the partition bounds are known.

        """
        return self._lower_bounds is not None \
            and self._upper_bounds is not None

    @property
    def known_sizes(self):
        """``True`` if the partition sizes (number of rows) are known.

        """
        return self._sizes is not None

    @property
    def sizes(self):
        """A tuple of integers holding the number of rows for each partition.

        ``None`` if the partition sizes are not known.

        """
        return self._sizes

    @property
    def lower_bounds(self):
        """A tuple holding the lower bounds for each partition.

        The length of ``lower_bounds`` is equal to the number of partitions.
        Each lower bound is a tuple with the same length as
        :py:attr:`padawan.Dataset.index_columns`.  Bounds are computed by
        constructing tuples from the index columns and using lexicographic
        ordering. So, a partition with index columns ``('a', 'b')`` and data

            = =
            a b  
            = =
            2 3
            1 2
            2 1
            = =

        will have the lower bound ``(1, 2)`` since the second row would become
        the first when the data is sorted by (`a`, `b`).

        """
        return self._lower_bounds

    @property
    def upper_bounds(self):
        """A tuple holding the lower bounds for each partition.

        The length of ``upper_bounds`` is equal to the number of partitions.
        Each upper bound is a tuple with the same length as
        :py:attr:`padawan.Dataset.index_columns`.  Bounds are computed by
        constructing tuples from the index columns and using lexicographic
        ordering. So, a partition with index columns ``('a', 'b')`` and data

            = =
            a b  
            = =
            2 1
            1 3
            1 2
            = =

        will have the upper bound ``(2, 1)`` since the first row would become
        the last when the data is sorted by (`a`, `b`).

        """
        return self._upper_bounds

    @property
    def known_schema(self):
        """``True`` if the table schema is known.

        """
        return self._schema is not None

    @property
    def schema(self):
        """A dict mapping column names to ``polars`` data types.

        Will be ``None`` if the schema is not known.

        """
        if self._schema is not None:
            return self._schema.copy()
        return None

    def is_disjoint(self):
        """Check if index ranges of partitions are non-overlapping.

        Returns:
          bool: ``True`` if index ranges do not overlap.
        """
        if self._npartitions <= 1:
            return True
        if not self._index_columns:
            return True
        if not self.known_bounds:
            raise StatsUnknownError(
                'Bounds must be known to check disjointness. '
                'Try using reindex first.'
            )

        partition_indices = sort_partitions(
            self._lower_bounds, self._upper_bounds)
        lower_bounds = [
            lex_key(self._lower_bounds[i]) for i in partition_indices]
        upper_bounds = [
            lex_key(self._upper_bounds[i]) for i in partition_indices]
        return all(
            ub < lb for ub, lb in zip(upper_bounds[:-1], lower_bounds[1:])
        )

    def assert_disjoint(self):
        """Assert that index ranges of partitions are non-overlapping.

        Raises:
          AssertionError: Index ranges overlap.

        Returns:
          padawan.Dataset: `self`.
        """
        if not self.is_disjoint():
            raise AssertionError('Partitions are not disjoint.')
        return self

    def __len__(self):
        return self._npartitions

    def _get_partition(self, partition_index):
        """Get a partition of the dataset.

        Args:
            partition_index (int): The index of the partition.

        Returns:
          polars.LazyFrame: The partition data.
        """
        raise NotImplementedError

    def __getitem__(self, partition_index):
        orig_index = partition_index
        if partition_index < 0:
            partition_index = self._npartitions + partition_index
        if partition_index >= self._npartitions or partition_index < 0:
            raise IndexError(f'Partition index {orig_index} is out of range.')
        return self._get_partition(partition_index)

    def _get_greedy(self, partition_index):
        return self[partition_index].collect()

    def __iter__(self):
        for i in range(self._npartitions):
            yield self[i]

    def _get_partition_with_stats(self, partition_index, index_columns=None):
        """Get a partition and the associated statistics.

        Args:
          partition_index (int): The index of the partition.
          index_columns (tuple of str, optional): The index columns to use
            for computing the stats. Defaults to ``None``, in which case
            ``self.index_columns`` is used.

        Returns:
          tuple: A tuple with the following components:

            `part` (polars.LazyFrame)
              The partition data.
            `nrows` (int)
              The number of rows in the partition.
            `lb` (tuple)
              The lower bound of the partiton.
              (One element for each index column.)
            `ub` (tuple)
              The upper bound of the partition.
              (One element for each index column.)
        """
        if index_columns is None:
            index_columns = self._index_columns
        else:
            index_columns = tuple(index_columns)

        part = self[partition_index].collect()
        nrows, lb, ub = get_partition_stats(part, index_columns)
        return part, nrows, lb, ub

    def _write_partition(self, partition_index, path, offset):
        fmt = f'part{{0:0>{PARTITION_NUMBER_DIGITS}d}}.parquet'
        filename = fmt.format(partition_index + offset)
        if self.known_sizes and self.known_bounds:
            nrows = self._sizes[partition_index]
            lb = self._lower_bounds[partition_index]
            ub = self._upper_bounds[partition_index]
            part = self[partition_index].collect()
        else:
            part, nrows, lb, ub \
                = self._get_partition_with_stats(partition_index)

        schema = None
        if partition_index == 0:
            schema = part.schema

        if nrows == 0:
            return None, 0, None, None, schema
        part.write_parquet(os.path.join(path, filename))
        return filename, nrows, lb, ub, schema

    def write_parquet(
            self,
            path,
            append=False,
            parallel=False,
            progress=False,
    ):
        """Write the dataset to disk.

        Args:
          path (str): The directory that will contain the parquet files.
            If a file or directory with that name exists (and `append` is
            ``False``) it will be deleted first.
          append (bool, optional): If ``True`` the partitions are appended to
            an existing dataset in `path`. In this case you must make sure that
            `path` is an existing directory containing valid padawan metadata
            files (e.g. by writing to the same path with
            ``write_parquet(path, append=False)`` first).
          parallel (bool or int): Specifies how to parallelize the computation:

              ``parallel = True``
                use all available CPUs
              ``parallel = False``
                no parallelism
              ``parallel > 1``
                use ``parallel`` number of CPUs
              ``parallel in [0, 1]``
                no parallelism
              ``parallel = -n < 0``
                use number of available CPUs minus n
          progress (callable, str, int, bool or tuple, optional):
            Whether to print and how to print progress messages about the
            computation. The possible values are:

              ``False``
                No progress messages are printed.
              ``True``
                A default progress message is printed after each completed
                partition.
              format string
                A custom message is printed after each completed partition.
                The following format variables can be used:

                  `completed` (int)
                    The number of completed partitions.
                  `total` (int)
                    The total number of partitions.
                  `remaining` (int)
                    The remaining number of partitions.
                  `start` (str)
                    The starting time of the computation in ISO format.
                  `finish` (str)
                    The expected finishing time of the computation in ISO 
                    format.
                  `telapsed` (str)
                    The elapsed time of the computations.
                  `tremaining` (str)
                    The expected time to finish.
                  `ttotal` (str)
                    The expected total time of the computation.
              integer `n`
                Print the default message only after every `n` completed
                partitions.
              tuple of the form ``(msg, n)``
                Print a custom message after every `n` completed partitions.
              callable
                Call a custom function after every completed partition. The
                function must accept the following arguments:

                  `completed` (int)
                    The number of completed partitions.
                  `total` (int)
                    The total number of partitions.
                  `start` (datetime.datetime)
                    The starting time of the computation.
                  `finish` (datetime.datetime)
                    The expected finishing time of the computation.

        Returns:
          padawan.Dataset or None: ``None`` if ``append=False``. Otherwise the
            dataset that was written is returned, as if it was read back in
            with :py:func:`padawan.scan_parquet`.

        """
        if not append:
            try:
                shutil.rmtree(path)
            except NotADirectoryError:
                os.remove(path)
            except FileNotFoundError:
                pass
            os.mkdir(path)
            files = []
            sizes = []
            lower_bounds = []
            upper_bounds = []
            max_partition_index = -1
            schema = None
        else:
            try:
                (
                    files,
                    index_columns,
                    sizes,
                    lower_bounds,
                    upper_bounds,
                    max_partition_index,
                    schema,
                ) = load_metadata(path)
            except FileNotFoundError:
                raise AppendError(
                    f'Could not load metadata in {repr(path)}.')
            if index_columns != self.index_columns:
                raise AppendError(
                    f'Cannot append dataset with index columns {index_columns}'
                    f' to dataset with index columns {self.index_columns}.')

        partition_indices = list(range(self._npartitions))
        meta = parallel_map(
            self._write_partition,
            partition_indices,
            workers=parallel,
            shared_args={'path': path, 'offset': max_partition_index + 1},
            progress=progress,
        )
        max_partition_index += self._npartitions

        files += [m[0] for m in meta if m[1] > 0]
        sizes += [m[1] for m in meta if m[1] > 0]
        lower_bounds += [m[2] for m in meta if m[1] > 0]
        upper_bounds += [m[3] for m in meta if m[1] > 0]
        if schema is None:
            if meta:
                schema = meta[0][4]
            else:
                schema = self._schema

        meta = {
            'index_columns': self._index_columns,
            'files': files,
            'sizes': sizes,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'max_partition_index': max_partition_index,
        }
        write_json(meta, os.path.join(path, METADATA_FILE))
        dataframe_from_schema(schema).write_parquet(
            os.path.join(path, SCHEMA_FILE))

        if append:
            return None
        return self._read_persisted(path)

    def collect(self, parallel=False, progress=False):
        """Pull all data into memory.

        Args:
          parallel (bool or int, optional): Specifies how to parallelize the
            computation. See corresponding argument for
            :py:meth:`padawan.Dataset.write_parquet` for details.
            Defaults to ``False``.
          progress (callable, str, int, bool or tuple, optional):
            Whether and how to print progress messages. See corresponding
            argument for :py:meth:`padawan.Dataset.write_parquet` for details.
            Defaults to ``False``.

        Returns:
          polars.DataFrame: A single dataframe with all partitions
            concatenated.
        """
        if self._npartitions == 0:
            return dataframe_from_schema(self._schema)

        partition_indices = list(range(self._npartitions))
        parts = parallel_map(
            self._get_greedy,
            partition_indices,
            workers=parallel,
            progress=progress,
        )
        return pl.concat(parts)
