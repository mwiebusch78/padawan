import pytest
import padawan
import polars as pl

from fixtures import datetime_sample
from utils import dataframe_eq


def test__from_polars__with_index_columns(datetime_sample):
    ds = padawan.from_polars(
        datetime_sample['data'],
        index_columns=['date', 'hour', 't'],
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == (len(datetime_sample['data']),)
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds'][:1]
    assert ds.upper_bounds == datetime_sample['upper_bounds'][-1:]
    assert ds.schema == datetime_sample['data'].schema
    assert dataframe_eq(ds.collect(), datetime_sample['data'])


def test__from_polars__without_index_columns(datetime_sample):
    ds = padawan.from_polars(datetime_sample['data'])
    assert ds.index_columns == ()
    assert ds.known_sizes is True
    assert ds.sizes == (len(datetime_sample['data']),)
    assert ds.known_bounds is True
    assert ds.lower_bounds == ((),)
    assert ds.upper_bounds == ((),)
    assert ds.schema == datetime_sample['data'].schema
    assert dataframe_eq(ds.collect(), datetime_sample['data'])
