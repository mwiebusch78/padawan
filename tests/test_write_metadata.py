import pytest
import padawan
import polars as pl
import datetime

from fixtures import (
    datetime_sample,
    output_dir,
)
from utils import dataframe_from_schema, dataframe_eq


def test__write_metadata(datetime_sample):
    padawan.write_metadata(
        datetime_sample['path'], 
        datetime_sample['index_columns'],
    )
    ds = padawan.scan_parquet(datetime_sample['path'])
    assert ds.index_columns == datetime_sample['index_columns']
    assert ds.sizes == datetime_sample['sizes']
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']
