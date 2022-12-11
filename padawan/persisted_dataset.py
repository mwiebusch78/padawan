import glob
import os
import polars as pl

from .dataset import Dataset, METADATA_FILE
from .parallelize import parallel_map
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

    def _scan_partition(self, path, index_columns):
        part = pl.scan_parquet(path)
        if index_columns:
            part = part.select(index_columns).collect()
            nrows = len(part)
            lb = min(part.rows())
            ub = max(part.rows())
        else:
            nrows = part.select(pl.count()).collect().row(0)[0]
            lb = ()
            ub = ()
        filename = os.path.basename(path)

        return filename, nrows, lb, ub

    def _scan_folder(self, path, index_columns, parallel):
        path_pattern = os.path.join(path, '*.parquet')
        partition_paths = sorted(glob.glob(path_pattern))
        if not partition_paths:
            raise FileNotFoundError(
                f'No files matching pattern {path_pattern}')

        meta = parallel_map(
            self._scan_partition,
            partition_paths,
            workers=parallel,
            shared_args={'index_columns': index_columns},
        )
        files = [m[0] for m in meta]
        sizes = [m[1] for m in meta]
        lower_bounds = [m[2] for m in meta]
        upper_bounds = [m[3] for m in meta]

        return files, sizes, lower_bounds, upper_bounds

    def _project_stats(self, stats, current_index_columns, index_columns):
        num_index_columns = len(index_columns)
        num_current_index_columns = len(current_index_columns)

        new_stats = []
        for obj in stats:
            new_stats.append(
                PartitionStats(
                    nrows=obj.nrows,
                    lb=obj.lb[:num_index_columns],
                    ub=obj.ub[:num_index_columns],
                )
            )
        return new_stats

    def __init__(self, path, index_columns=None, parallel=False):
        self._path = path

        try:
            files, current_index_columns, sizes, lower_bounds, upper_bounds \
                = self._load_metadata(path)
        except FileNotFoundError:
            if index_columns is None:
                raise StatsNotFoundError(
                    f'No stats found under {path}. '
                    'Specify index_columns to trigger a scan.')
            files, sizes, lower_bounds, upper_bounds \
                = self._scan_folder(path, index_columns, parallel)
            current_index_columns = tuple(index_columns)

        if index_columns is not None:
            index_columns = tuple(index_columns)
            num_index_columns = len(index_columns)
            num_current_index_columns = len(current_index_columns)
            if not (num_index_columns <= num_current_index_columns
                    and index_columns == 
                    current_index_columns[:num_index_columns]):
                files, sizes, lower_bounds, upper_bounds \
                    = self._scan_folder(path, index_columns, parallel)
            else:
                lower_bounds = [b[:num_index_columns] for b in lower_bounds]
                upper_bounds = [b[:num_index_columns] for b in upper_bounds]
        else:
            index_columns = current_index_columns

        super().__init__(
            len(files), index_columns, sizes, lower_bounds, upper_bounds)
        self._files = files

    def __getitem__(self, partition_index):
        path = os.path.join(self._path, self._files[partition_index])
        part = pl.scan_parquet(path)
        return part


def read_parquet(path, index_columns=None, parallel=False):
    return PersistedDataset(
        path, index_columns=index_columns, parallel=parallel)


def _read_persisted(self, path):
    return read_parquet(path)
Dataset._read_persisted = _read_persisted
