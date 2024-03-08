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


def test__slice__irrelevant_bounds(datetime_sample):
    lb = (None, timedelta(hours=-10))
    ub = (date(2022, 1, 6), timedelta(hours=10))
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub)
    )

    assert ds.lower_bounds == \
        tuple(b[:2] for b in datetime_sample['lower_bounds'])
    assert ds.upper_bounds == \
        tuple(b[:2] for b in datetime_sample['upper_bounds'])
    assert ds.sizes == datetime_sample['sizes']


def test__slice__inclusive_none(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=23))
    ub = (date(2022, 1, 4), timedelta(hours=0))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub, inclusive='none')
    )

    assert len(ds) == 1

    ds = ds.collect().select(['date', 'hour']).sort(['date', 'hour'])
    assert(ds.row(0) > lb)
    assert(ds.row(-1) < ub)


def test__slice__inclusive_lower(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=23))
    ub = (date(2022, 1, 4), timedelta(hours=0))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub, inclusive='lower')
    )

    assert len(ds) == 2

    ds = ds.collect().select(['date', 'hour']).sort(['date', 'hour'])
    assert(ds.select(['date', 'hour']).row(0) == lb)
    assert(ds.select(['date', 'hour']).row(-1) < ub)


def test__slice__inclusive_upper(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=23))
    ub = (date(2022, 1, 4), timedelta(hours=0))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub, inclusive='upper')
    )

    assert len(ds) == 2

    ds = ds.collect().select(['date', 'hour']).sort(['date', 'hour'])
    assert(ds.select(['date', 'hour']).row(0) > lb)
    assert(ds.select(['date', 'hour']).row(-1) == ub)


def test__slice__inclusive_both(datetime_sample):
    lb = (date(2022, 1, 2), timedelta(hours=23))
    ub = (date(2022, 1, 4), timedelta(hours=0))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub, inclusive='both')
    )

    assert len(ds) == 3

    ds = ds.collect().select(['date', 'hour']).sort(['date', 'hour'])
    assert(ds.select(['date', 'hour']).row(0) == lb)
    assert(ds.select(['date', 'hour']).row(-1) == ub)


def test__slice__firstcol(datetime_sample):
    lb = [date(2022, 1, 2)]
    ub = [date(2022, 1, 4)]

    data = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'])
        .slice(lb, ub, inclusive='both')
        .collect()
    )
    expected_data = (
        datetime_sample['data']
        .filter(
            (pl.col('date') >= lb[0])
            & (pl.col('date') <= ub[0])
        )
    )
    assert dataframe_eq(data, expected_data)
