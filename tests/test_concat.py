import pytest
import padawan
import polars as pl
from datetime import date, datetime, timedelta

from fixtures import datetime_sample
from utils import dataframe_eq


def test__concat(datetime_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
    )
    ds = padawan.concat([ds1, ds1])

    assert len(ds) == 2*len(ds1)
    assert ds.sizes == ds1.sizes*2
    assert ds.lower_bounds == ds1.lower_bounds*2
    assert ds.upper_bounds == ds1.upper_bounds*2

    expected_ds = pl.concat(
        [datetime_sample['data'], datetime_sample['data']])
    assert dataframe_eq(expected_ds, ds.collect())


def test__concat__with_empty(datetime_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
    )
    ds2 = padawan.concat([])
    ds = padawan.concat([ds1, ds2])

    assert len(ds) == len(ds1)
    assert ds.sizes == ds1.sizes
    assert ds.lower_bounds == ds1.lower_bounds
    assert ds.upper_bounds == ds1.upper_bounds

    assert dataframe_eq(datetime_sample['data'], ds.collect())


