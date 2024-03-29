from .dataset import get_partition_stats, dataframe_from_schema
from .parallelize import parallel_map
from .metadata import METADATA_FILE, SCHEMA_FILE
from .persisted_dataset import scan_folder
from .json_io import write_json

import polars as pl
import os


def _get_file_stats(file_index_and_name, dirname, index_columns):
    file_index, file_name = file_index_and_name
    path = os.path.join(dirname, file_name)
    part = pl.read_parquet(path)
    nrows, lb, ub = get_partition_stats(part, index_columns)
    schema = None
    if file_index == 0:
        schema = part.schema
    return file_name, nrows, lb, ub, schema


def write_metadata(
        path,
        index_columns,
        parallel=False,
        progress=False,
):
    """Add or overwrite metadata in existing folder.

    This function can be used on directories containing multiple parquet
    files which were not generated by padawan. It scans the files and adds
    the necessary metadata files to the directory. After that partition sizes
    and bounds will be known straight away when the directory
    is read with :py:func:`padawan.scan_parquet` (without the need to do a
    :py:meth:`padawan.Dataset.reindex`).

    Args:
      path (str): Path to the directory to be scanned.
      index_columns (tuple of str): The index columns to use
        for computing the metadata.
      parallel (bool or int, optional): Specifies how to parallelize the
        computation. See corresponding argument for
        :py:meth:`padawan.Dataset.write_parquet` for details.
        Defaults to ``False``.
      progress (callable, str, int, bool or tuple, optional):
        Whether and how to print progress messages. See corresponding
        argument for :py:meth:`padawan.Dataset.write_parquet` for details.
        Defaults to ``False``.
    """
    index_columns = tuple(index_columns)
    files = scan_folder(path)
    args = list(enumerate(files))
    meta = parallel_map(
        _get_file_stats,
        args,
        workers=parallel,
        shared_args={'dirname': path, 'index_columns': index_columns},
        progress=progress,
    )

    files = [m[0] for m in meta if m[1] > 0]
    sizes = [m[1] for m in meta if m[1] > 0]
    lower_bounds = [m[2] for m in meta if m[1] > 0]
    upper_bounds = [m[3] for m in meta if m[1] > 0]
    schema = None
    if meta:
        schema = meta[0][4]

    meta = {
        'index_columns': index_columns,
        'files': files,
        'sizes': sizes,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'max_partition_index': len(files) - 1,
    }
    write_json(meta, os.path.join(path, METADATA_FILE))
    dataframe_from_schema(schema).write_parquet(
        os.path.join(path, SCHEMA_FILE))
