import pytest
import padawan
import polars as pl
import numpy as np

from fixtures import (
    datetime_sample,
    output_dir,
)


def test__map__preserves_none(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(lambda df: df.with_columns((2*pl.col('a')).alias('b')))
    )
    assert ds.known_sizes is False
    assert ds.known_bounds is False

    df = ds.collect()
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a)


def test__map__preserves_bounds(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(
            lambda df: df.with_columns((2*pl.col('a')).alias('b')),
            preserves='bounds',
        )
    )
    assert ds.known_sizes is False
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    df = ds.collect()
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a)


def test__map__preserves_sizes(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(
            lambda df: df.with_columns((2*pl.col('a')).alias('b')),
            preserves='sizes',
        )
    )
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is False

    df = ds.collect()
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a)


def test__map__preserves_all(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(
            lambda df: df.with_columns((2*pl.col('a')).alias('b')),
            preserves='all',
        )
    )
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    df = ds.collect()
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a)


def test__map__sequential_with_args(datetime_sample):
    def f(df, alpha, beta):
        return df.with_columns((alpha*pl.col('a') + beta).alias('b'))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(
            f,
            shared_args=[2, 3],
            preserves='all',
        )
    )
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    df = ds.collect()
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a + 3)


def test__map__parallel_with_args(datetime_sample):
    def f(df, alpha, beta):
        return df.with_columns((alpha*pl.col('a') + beta).alias('b'))

    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .map(
            f,
            shared_args=[2, 3],
            preserves='all',
        )
    )
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    df = ds.collect(parallel=2)
    b = df.get_column('b').to_numpy()
    a = datetime_sample['data'].get_column('a').to_numpy()
    assert np.all(b == 2*a + 3)
