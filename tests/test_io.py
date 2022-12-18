import pytest
import padawan

from fixtures import (
    datetime_sample,
    output_dir,
)


def test__scan_parquet__with_index_columns(datetime_sample):
    ds = padawan.scan_parquet(
        datetime_sample['path'],
        index_columns=['date', 'hour'],
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
    assert ds.known_bounds is False
    assert ds.lower_bounds is None
    assert ds.upper_bounds is None


def test__collect_stats__sequential(datetime_sample):
    ds = (
        padawan.scan_parquet(
            datetime_sample['path'],
            index_columns=['date', 'hour', 't'],
        )
        .collect_stats()
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__collect_stats__parallel(datetime_sample):
    ds = (
        padawan.scan_parquet(
            datetime_sample['path'],
            index_columns=['date', 'hour', 't'],
        )
        .collect_stats(parallel=2)
    )
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == datetime_sample['sizes']
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds']
    assert ds.upper_bounds == datetime_sample['upper_bounds']


def test__write_parquet__sequential(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(
            datetime_sample['path'],
            index_columns=['date', 'hour', 't'],
        )
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
        padawan.scan_parquet(
            datetime_sample['path'],
            index_columns=['date', 'hour', 't'],
        )
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

