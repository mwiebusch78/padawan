import shutil
import os
import polars as pl

from .parallelize import parallel_map, is_parallel_config
from .json_io import write_json


PARTITION_NUMBER_DIGITS = 10
METADATA_FILE = '_padawan_metadata.json'
SCHEMA_FILE = '_padawan_schema'


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


class StatsUnknownError(Exception):
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
        nrows = len(part)
        if index_columns:
            index = part.select(index_columns)
            lb = lex_min(index)
            ub = lex_max(index)
        else:
            lb = ()
            ub = ()
        return part, nrows, lb, ub

    def _write_partition(self, partition_index, path):
        fmt = f'part{{0:0>{PARTITION_NUMBER_DIGITS}d}}.parquet'
        filename = fmt.format(partition_index)
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

    def write_parquet(self, path, parallel=False):
        """Write the dataset to disk.

        Args:
          path (str): The directory that will contain the parquet files.
            If a file or directory with that name exists it will be deleted
            first.
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

        Returns:
          padawan.Dataset: The dataset that was written, as if it
            was read back in with :py:func:`padawan.scan_parquet`.

        """
        try:
            shutil.rmtree(path)
        except NotADirectoryError:
            os.remove(path)
        except FileNotFoundError:
            pass
        os.mkdir(path)

        partition_indices = list(range(self._npartitions))
        meta = parallel_map(
            self._write_partition,
            partition_indices,
            workers=parallel,
            shared_args={'path': path},
        )

        files = [m[0] for m in meta if m[1] > 0]
        sizes = [m[1] for m in meta if m[1] > 0]
        lower_bounds = [m[2] for m in meta if m[1] > 0]
        upper_bounds = [m[3] for m in meta if m[1] > 0]
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
        }
        write_json(meta, os.path.join(path, METADATA_FILE))
        dataframe_from_schema(schema).write_parquet(
            os.path.join(path, SCHEMA_FILE))

        return self._read_persisted(path)

    def collect(self, parallel=False):
        """Pull all data into memory.

        Args:
          parallel (bool or int, optional): Specifies how to parallelize the
            computation:

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

            Defaults to ``False``.

        Returns:
          polars.DataFrame: A single dataframe with all partitions
            concatenated.
        """
        if self._npartitions == 0:
            return dataframe_from_schema(self._schema)

        partition_indices = list(range(self._npartitions))
        if is_parallel_config(parallel):
            parts = parallel_map(
                self._get_greedy,
                partition_indices,
                workers=parallel,
            )
            return pl.concat(parts)
        else:
            parts = [self[i] for i in partition_indices]
            return pl.concat(parts).collect()
