import pytest
import padawan

from fixtures import (
    datetime_sample,
    output_dir,
)


def test__collate__sequential(datetime_sample, output_dir):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date', 'hour', 't'])
        .collate(48)
    )

    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == (50, 48)
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds'][0::2]
    assert ds.upper_bounds == datetime_sample['upper_bounds'][1::2]

    ds = ds.write_parquet(output_dir)
    assert ds.index_columns == ('date', 'hour', 't')
    assert ds.known_sizes is True
    assert ds.sizes == (50, 48)
    assert ds.known_bounds is True
    assert ds.lower_bounds == datetime_sample['lower_bounds'][0::2]
    assert ds.upper_bounds == datetime_sample['upper_bounds'][1::2]

