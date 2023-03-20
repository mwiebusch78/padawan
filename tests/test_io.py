import pytest
import padawan
import polars as pl
import datetime

from fixtures import (
    datetime_sample,
    output_dir,
)
from utils import dataframe_from_schema, dataframe_eq


def test__scan_parquet__with_index_columns(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour'], collect_stats=False)
    )
    assert ds.index_columns == ('date', 'hour')
    assert ds.known_sizes is False
    assert ds.sizes is None
    assert ds.known_bounds is False
    assert ds.lower_bounds is None
    assert ds.upper_bounds is None


def test__scan_parquet__without_index_columns(datetime_sample):
    ds = padawan.scan_parquet(datetime_sample['path'])
    assert ds.index_columns == ()
    assert ds.known_sizes is False
    assert ds.sizes is None
    assert ds.known_bounds is True
    assert ds.lower_bounds == ((),)*len(ds)
    assert ds.upper_bounds == ((),)*len(ds)


def test__reindex__sequential(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__reindex__parallel(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'], parallel=2)
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__reindex__no_index_cols(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex()
    )
    assert ds.index_columns == ()
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == ((),)*len(ds)
    assert ds.upper_bounds == ((),)*len(ds)


def test__write_parquet__sequential(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'], collect_stats=False)
        .write_parquet(output_dir)
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    ds = padawan.scan_parquet(output_dir)
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__write_parquet__parallel(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'], collect_stats=False)
        .write_parquet(output_dir, parallel=2)
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']

    ds = padawan.scan_parquet(output_dir)
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__write_parquet__append(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
    )
    ds1 = (
        ds
        .slice(ub=(datetime.date(2022, 1, 3), None, None), inclusive='lower')
        .write_parquet(output_dir)
    )
    (
        ds
        .slice(lb=(datetime.date(2022, 1, 3), None, None), inclusive='lower')
        .write_parquet(output_dir, append=True)
    )

    ds = padawan.scan_parquet(output_dir)
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__write_parquet__append_to_empty(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
    )
    ds1 = (
        ds
        .slice(lb=(datetime.date(2022, 1, 10), None, None), inclusive='lower')
        .write_parquet(output_dir)
    )
    ds.write_parquet(output_dir, append=True)

    ds = padawan.scan_parquet(output_dir)
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__write_parquet__empty(datetime_sample, output_dir):
    schema = {'a': pl.Int64, 'b': pl.Float64}
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .map(lambda part: dataframe_from_schema(schema))
        .reindex(['a'])
        .write_parquet(output_dir)
    )
    assert len(ds) == 0
    assert ds.schema == schema
    assert ds.collect().schema == schema


def test__write_parquet__empty_unknown_schema(datetime_sample, output_dir):
    schema = {'a': pl.Int64, 'b': pl.Float64}
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .map(lambda part: dataframe_from_schema(schema))
        .write_parquet(output_dir)
    )
    assert len(ds) == 0
    assert ds.schema == schema
    assert ds.collect().schema == schema


def test__write_parquet__empty_unknown_stats(datetime_sample, output_dir):
    schema = {'a': pl.Int64, 'b': pl.Float64}
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .map(lambda part: dataframe_from_schema(schema))
        .reindex(['a'], collect_stats=False)
        .write_parquet(output_dir)
    )
    assert len(ds) == 0
    assert ds.schema == schema
    assert ds.collect().schema == schema


def test__collect__sequential(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .collect()
    )
    assert all(
        (ds == datetime_sample['data'])
        .select(pl.col('*').all()).row(0)
    )


def test__collect__parallel(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .collect(parallel=2)
    )
    assert all(
        (ds == datetime_sample['data'])
        .select(pl.col('*').all()).row(0)
    )


def test__iter(datetime_sample):
    ds = padawan.scan_parquet(datetime_sample['path'])
    for i in range(4):
        part = ds[2*i].collect()
        start = datetime_sample['divisions'][i]
        end = datetime_sample['divisions'][i+1]
        comp = datetime_sample['data'][start:end, :]
        assert all(
            (part == comp).select(pl.col('*').all()).row(0)
        )
        part = ds[2*i+1].collect()
        assert len(part) == 0

