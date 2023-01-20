import pytest
import padawan
import polars as pl
from datetime import date, datetime, timedelta

from fixtures import datetime_sample
from utils import dataframe_eq


def test__rename(datetime_sample):
    mapping = {
        'date': 'date_2',
        'hour': 'hour_2',
        'a': 'a_2',
    }
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .rename(mapping)
    )
    expected_ds = datetime_sample['data'].rename(mapping)

    assert ds.index_columns == ('date_2', 'hour_2', 't')
    assert dataframe_eq(ds.collect(), expected_ds)
    assert ds.schema == expected_ds.schema
