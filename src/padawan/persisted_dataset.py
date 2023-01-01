import glob
import os
import polars as pl

from .dataset import Dataset, METADATA_FILE
from .json_io import read_json


class StatsNotFoundError(Exception):
    pass


class PersistedDataset(Dataset):
    def _load_metadata(self, path):
        meta = read_json(os.path.join(path, METADATA_FILE))
    
        index_columns = tuple(meta['index_columns'])
        files = meta['files']
        sizes = meta['sizes']
        lower_bounds = [tuple(b) for b in meta['lower_bounds']]
        upper_bounds = [tuple(b) for b in meta['upper_bounds']]

        assert len(files) == len(sizes)
        assert len(files) == len(lower_bounds)
        assert len(files) == len(upper_bounds)
        return files, index_columns, sizes, lower_bounds, upper_bounds

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
            files, index_columns, sizes, lower_bounds, upper_bounds \
                = self._load_metadata(path)
        except FileNotFoundError:
            files = self._scan_folder(path)
            sizes = None
            lower_bounds = ((),)*len(files)
            upper_bounds = ((),)*len(files)
            index_columns = tuple()

        super().__init__(
            len(files), index_columns, sizes, lower_bounds, upper_bounds)
        self._files = files

    def _get_partition(self, partition_index):
        path = os.path.join(self._path, self._files[partition_index])
        part = pl.scan_parquet(path)
        return part


def _read_persisted(self, path):
    return PersistedDataset(path)
Dataset._read_persisted = _read_persisted


def scan_parquet(path):
    return PersistedDataset(path)
