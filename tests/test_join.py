import pytest
import padawan
import polars as pl
from datetime import date, datetime, timedelta

from fixtures import datetime_sample, date_sample
from utils import dataframe_eq


def test__join__inner(datetime_sample, date_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date'])
    )
    ds2 = (
        padawan.scan_parquet(date_sample['path'])
        .reindex(['date'])
    )
    ds = ds1.join(ds2, how='inner').collect()

    expected_ds = datetime_sample['data'].join(
        date_sample['data'], on='date', how='inner')
    assert dataframe_eq(expected_ds, ds)


def test__join__left(datetime_sample, date_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date'])
    )
    ds2 = (
        padawan.scan_parquet(date_sample['path'])
        .reindex(['date'])
    )
    ds = ds1.join(ds2, how='left').collect()

    expected_ds = datetime_sample['data'].join(
        date_sample['data'], on='date', how='left')
    assert dataframe_eq(expected_ds, ds)


def test__join__outer(datetime_sample, date_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date'])
    )
    ds2 = (
        padawan.scan_parquet(date_sample['path'])
        .reindex(['date'])
    )
    ds = ds1.join(ds2, how='outer').collect()

    expected_ds = datetime_sample['data'].join(
        date_sample['data'], on='date', how='outer')
    assert dataframe_eq(expected_ds, ds)

