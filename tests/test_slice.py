import pytest
import padawan
import polars as pl
from datetime import date, datetime, timedelta


from fixtures import datetime_sample
from utils import dataframe_eq


def test__slice__single_partition(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=6))
    ub = (date(2022, 1, 2), timedelta(hours=18))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    assert len(ds) == 1

    data = ds.collect()
    expected_data = (
        datetime_sample['data']
        .filter(pl.col('t') >= datetime(2022, 1, 2, 6))
        .filter(pl.col('t') < datetime(2022, 1, 2, 18))
    )
    assert dataframe_eq(data, expected_data)

def test__slice__two_partitions(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=6))
    ub = (date(2022, 1, 3), timedelta(hours=18))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    assert len(ds) == 2

    data = ds.collect()
    expected_data = (
        datetime_sample['data']
        .filter(pl.col('t') >= datetime(2022, 1, 2, 6))
        .filter(pl.col('t') < datetime(2022, 1, 3, 18))
    )
    assert dataframe_eq(data, expected_data)


def test__slice__null_bounds(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(None, None)
    )

    data = ds.collect()
    expected_data = datetime_sample['data']
    assert dataframe_eq(data, expected_data)


def test__slice__no_upper_bound(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=6))
    ub = None
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    data = ds.collect()
    expected_data = (
        datetime_sample['data']
        .filter(pl.col('t') >= datetime(2022, 1, 2, 6))
    )
    assert dataframe_eq(data, expected_data)


def test__slice__no_lower_bound(datetime_sample):
    lb = None
    ub = (date(2022, 1, 3), timedelta(hours=18))
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    data = ds.collect()
    expected_data = (
        datetime_sample['data']
        .filter(
            pl.col('t').is_null()
            | (pl.col('t') < datetime(2022, 1, 3, 18))
        )
    )
    assert dataframe_eq(data, expected_data)


def test__slice__null_in_bounds(datetime_sample):
    lb = (None, timedelta(hours=-10))
    ub = (date(2022, 1, 1), None)
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    assert len(ds) == 1

    data = ds.collect()
    expected_data = (
        datetime_sample['data']
        .filter(pl.col('date').is_null())
    )
    assert dataframe_eq(data, expected_data)


def test__slice__flipped_bounds(datetime_sample):
    lb = (date(2022, 1, 1), None)
    ub = (None, timedelta(hours=-10))
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    assert len(ds) == 0

    data = ds.collect()
    assert len(data) == 0
