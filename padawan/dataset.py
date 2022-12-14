import shutil
import os
import polars as pl

from .parallelize import parallel_map
from .json_io import write_json


PARTITION_NUMBER_DIGITS = 10
METADATA_FILE = '_dpart_metadata.json'


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
        self._sizes = None if sizes is None else [int(s) for s in sizes]
        self._lower_bounds = None if lower_bounds is None \
            else [tuple(b) for b in lower_bounds]
        self._upper_bounds = None if upper_bounds is None \
            else [tuple(b) for b in upper_bounds]

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
        return None if self._sizes is None else list(self._sizes)

    @property
    def lower_bounds(self):
        return None if self._lower_bounds is None else list(self._lower_bounds)

    @property
    def upper_bounds(self):
        return None if self._upper_bounds is None else list(self._upper_bounds)

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
            lb = min(index.rows())
            ub = max(index.rows())
        else:
            lb = ()
            ub = ()
        return part, nrows, lb, ub

    def _get_partition_stats(self, partition_index):
        _, nrows, lb, ub = self._get_partition_with_stats(partition_index)
        return nrows, lb, ub

    def _compute_stats(self, parallel):
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
        if self.known_sizes and self.known_bounds:
            return
        partition_indices = list(range(self._npartitions))
        stats = parallel_map(
            self._get_partition_stats,
            partition_indices,
            workers=parallel,
        )
        self._sizes = [s[0] for s in stats]
        self._lower_bounds = [s[1] for s in stats]
        self._upper_bounds = [s[2] for s in stats]

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

        files = [m[0] for m in meta]
        sizes = [m[1] for m in meta]
        lower_bounds = [m[2] for m in meta]
        upper_bounds = [m[3] for m in meta]

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
        partition_indices = list(range(self._npartitions))
        parts = parallel_map(
            self.__getitem__,
            partition_indices,
            workers=parallel,
        )
        return pl.concat(parts).collect()

