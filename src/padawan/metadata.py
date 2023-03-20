import os
import polars as pl
from .json_io import read_json


PARTITION_NUMBER_DIGITS = 10
METADATA_FILE = '_padawan_metadata.json'
SCHEMA_FILE = '_padawan_schema'


def load_metadata(path):
    meta = read_json(os.path.join(path, METADATA_FILE))
    schema = pl.read_parquet(
        os.path.join(path, SCHEMA_FILE)).schema

    index_columns = tuple(meta['index_columns'])
    files = meta['files']
    sizes = meta['sizes']
    lower_bounds = [tuple(b) for b in meta['lower_bounds']]
    upper_bounds = [tuple(b) for b in meta['upper_bounds']]
    max_partition_index = meta['max_partition_index']

    assert len(files) == len(sizes)
    assert len(files) == len(lower_bounds)
    assert len(files) == len(upper_bounds)
    return (
        files,
        index_columns,
        sizes,
        lower_bounds,
        upper_bounds,
        max_partition_index,
        schema,
    )
