import pytest
import padawan
from datetime import datetime, timedelta

from padawan.repartitioned_dataset import (
    get_row_divisions,
    get_index_divisions,
)

from fixtures import (
    datetime_sample,
    output_dir,
)


def test__get_row_divisions__nothing_to_do():
    divisions, sizes, lower_bounds, upper_bounds \
        = get_row_divisions([5, 5, 5, 5], 5)
    expected_divisions = [(1, 0), (2, 0), (3, 0)]
    assert divisions == expected_divisions
    assert sizes == [5]*4
    assert lower_bounds == [()]*4
    assert upper_bounds == [()]*4


def test__get_row_divisions__with_tail():
    # 0 0 0 1 1 1 1 1 2 2 3 3 3 3 3 3 3
    # 0 1 2 0 1 2 3 4 0 1 0 1 2 3 4 5 6
    #           .         .         .
    divisions, sizes, lower_bounds, upper_bounds \
        = get_row_divisions([3, 5, 2, 7], 5)
    expected_divisions = [(1, 2), (3, 0), (3, 5)]
    assert divisions == expected_divisions
    assert sizes == [5, 5, 5, 2]
    assert lower_bounds == [()]*4
    assert upper_bounds == [()]*4


def test__get_row_divisions__without_tail():
    # 0 0 0 1 1 1 1 1 2 2 3 3 3 3 3
    # 0 1 2 0 1 2 3 4 0 1 0 1 2 3 4
    #           .         .        
    divisions, sizes, lower_bounds, upper_bounds \
        = get_row_divisions([3, 5, 2, 5], 5)
    expected_divisions = [(1, 2), (3, 0)]
    assert divisions == expected_divisions
    assert sizes == [5, 5, 5]
    assert lower_bounds == [()]*3
    assert upper_bounds == [()]*3


def test__get_index_divisions(datetime_sample):
    ds = padawan.scan_parquet(
        datetime_sample['path'],
        index_columns=['date'],
    )
    rows_per_partition = 24 
    sample_fraction = 1.0
    index_columns = ('hour',)
    base_seed = 10
    seed_increment = 10
    parallel = False

    divisions, sizes, lower_bounds, upper_bounds = get_index_divisions(
        ds,
        rows_per_partition,
        sample_fraction,
        index_columns,
        base_seed,
        seed_increment,
        parallel,
    )
    expected_divisions = [(timedelta(hours=h),) for h in range(6, 24, 6)]
    expected_sizes = [24]*4
    expected_lower_bounds = [(timedelta(hours=h),) for h in range(0, 24, 6)]
    expected_upper_bounds = [(timedelta(hours=h),) for h in range(5, 24, 6)]
    assert divisions == expected_divisions
    assert sizes == expected_sizes
    assert lower_bounds == expected_lower_bounds
    assert upper_bounds == expected_upper_bounds

