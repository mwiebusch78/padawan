import glob
import os
import polars as pl

from .dataset import Dataset, METADATA_FILE, SCHEMA_FILE
from .json_io import read_json


class StatsNotFoundError(Exception):
    pass


class PersistedDataset(Dataset):
    def _load_metadata(self, path):
        meta = read_json(os.path.join(path, METADATA_FILE))
        schema = pl.read_parquet(
            os.path.join(path, SCHEMA_FILE)).schema
    
        index_columns = tuple(meta['index_columns'])
        files = meta['files']
        sizes = meta['sizes']
        lower_bounds = [tuple(b) for b in meta['lower_bounds']]
        upper_bounds = [tuple(b) for b in meta['upper_bounds']]

        assert len(files) == len(sizes)
        assert len(files) == len(lower_bounds)
        assert len(files) == len(upper_bounds)
        return files, index_columns, sizes, lower_bounds, upper_bounds, schema

    def _scan_folder(self, path):
        path_pattern = os.path.join(path, '*.parquet')
        partition_paths = sorted(glob.glob(path_pattern))
        if not partition_paths:
            raise FileNotFoundError(
                f'No files matching pattern {path_pattern}')

        return [os.path.basename(f) for f in partition_paths]

    def __init__(self, path):
        self._path = path

        try:
            files, index_columns, sizes, lower_bounds, upper_bounds, schema \
                = self._load_metadata(path)
        except FileNotFoundError:
            files = self._scan_folder(path)
            sizes = None
            lower_bounds = ((),)*len(files)
            upper_bounds = ((),)*len(files)
            index_columns = tuple()
            schema=None

        super().__init__(
            len(files),
            index_columns,
            sizes,
            lower_bounds,
            upper_bounds,
            schema,
        )
        self._files = files

    def _get_partition(self, partition_index):
        path = os.path.join(self._path, self._files[partition_index])
        part = pl.scan_parquet(path)
        return part


def _read_persisted(self, path):
    return PersistedDataset(path)
Dataset._read_persisted = _read_persisted


def scan_parquet(path):
    """Read partitioned data from disk.

    Args:
      path (str): Path to a directory holding the partitioned data. Each file
        under `path` ending in ``.parquet`` will become a partition of the
        dataset. Metadata about the partitions such as schema, partition
        sizes and bounds may be stored in special files named
        ``_padawan_metadata.json`` and ``_padawan_schema``. These files are
        created automatically when the data is written with
        :py:meth:`padawan.Dataset.write_parquet`. If they are not present
        the resulting :py:class:`padawan.Dataset` will have unknown bounds
        and sizes.

    Returns:
      padawan.Dataset: A dataset representing the data under `path`.

    """
    return PersistedDataset(path)
