import pytest
import padawan
import polars as pl
from datetime import date, datetime, timedelta

from fixtures import datetime_sample, date_sample
from utils import dataframe_eq


def test__combine(datetime_sample, date_sample):
    ds1 = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date'])
    )
    ds2 = (
        padawan.scan_parquet(date_sample['path'])
        .reindex(['date'])
    )

    def func(part1, part2):
        return part1.join(part2, on='date', how='inner')

    ds = padawan.combine([ds1, ds2], func).collect()

    expected_ds = datetime_sample['data'].join(
        date_sample['data'], on='date', how='inner')
    assert dataframe_eq(expected_ds, ds)


