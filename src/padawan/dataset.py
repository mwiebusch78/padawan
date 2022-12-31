import shutil
import os
import polars as pl

from .parallelize import parallel_map, is_parallel_config
from .json_io import write_json


PARTITION_NUMBER_DIGITS = 10
METADATA_FILE = '_padawan_metadata.json'


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


class StatsUnknownError(Exception):
    pass


class Dataset:
    def __init__(
            self,
            npartitions,
            index_columns=(),
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
    ):
        self._index_columns = tuple(index_columns)
        self._npartitions = npartitions

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
        return self._index_columns

    @property
    def known_bounds(self):
        return self._lower_bounds is not None \
            and self._upper_bounds is not None

    @property
    def known_sizes(self):
        return self._sizes is not None

    @property
    def sizes(self):
        return self._sizes

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    def __len__(self):
        return self._npartitions

    def __getitem__(self, partition_index):
        """Get a partition of the dataset.

        Args:
            partition_index (int): The index of the partition.

        Returns:
          part (polars.LazyFrame): The partition data.
        """
        raise NotImplementedError

    def _get_greedy(self, partition_index):
        return self[partition_index].collect()

    def __iter__(self):
        for i in range(self._npartitions):
            yield self[i]

    def _get_partition_with_stats(self, partition_index):
        """Get a partition and the associated statistics.

        Args:
            partition_index (int): The index of the partition.

        Returns:
          part (polars.LazyFrame): The partition data.
          nrows (int): The number of rows in the partition.
          lb (tuple): The lower bound of the partiton.
            (One element for each index column.)
          ub (tuple): The upper bound of the partition.
            (One element for each index column.)
        """
        part = self[partition_index].collect()
        nrows = len(part)
        if self._index_columns:
            index = part.select(self._index_columns)
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
        if nrows == 0:
            return None, 0, None, None
        part.write_parquet(os.path.join(path, filename))
        return filename, nrows, lb, ub

    def write_parquet(self, path, parallel=False):
        """Write the dataset to disk.

        Args:
          path (str): The directory that will contain the parquet files.
            If a file or directory with that name exists it will be deleted
            first.

        Kwargs:
          parallel (bool or int): Specifies how to parallelize computation:
            `parallel = True` -- use all available CPUs
            `parallel = False` -- no parallelism
            `parallel > 1` -- use `parallel` number of CPUs
            `parallel in [0, 1]` -- no parallelism
            `parallel = -n < 0` -- use number of available CPUs minus n
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

        meta = {
            'index_columns': self._index_columns,
            'files': files,
            'sizes': sizes,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
        }
        write_json(meta, os.path.join(path, METADATA_FILE))

        return self._read_persisted(path)

    def collect(self, parallel=False):
        """Pull all data into memory.

        Kwargs:
          parallel (bool or int): Specifies how to parallelize computation:
            `parallel = True` -- use all available CPUs
            `parallel = False` -- no parallelism
            `parallel > 1` -- use `parallel` number of CPUs
            `parallel in [0, 1]` -- no parallelism
            `parallel = -n < 0` -- use number of available CPUs minus n

        Returns:
          data (polars.DataFrame): A single dataframe with all partitions
            concatenated.
        """
        if self._npartitions == 0:
            return pl.DataFrame()
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
