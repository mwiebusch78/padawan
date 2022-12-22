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
    result = get_row_divisions([5, 5, 5, 5], 5)
    expected_result = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert result == expected_result


def test__get_row_divisions__with_tail():
    # 0 0 0 1 1 1 1 1 2 2 3 3 3 3 3 3 3
    # 0 1 2 0 1 2 3 4 0 1 0 1 2 3 4 5 6
    # .         .         .         .
    result = get_row_divisions([3, 5, 2, 7], 5)
    expected_result = [(0, 0), (1, 2), (3, 0), (3, 5)]
    assert result == expected_result


def test__get_row_divisions__without_tail():
    # 0 0 0 1 1 1 1 1 2 2 3 3 3 3 3
    # 0 1 2 0 1 2 3 4 0 1 0 1 2 3 4
    # .         .         .        
    result = get_row_divisions([3, 5, 2, 5], 5)
    expected_result = [(0, 0), (1, 2), (3, 0)]
    assert result == expected_result


def test__get_index_divisions(datetime_sample):
    ds = padawan.scan_parquet(
        datetime_sample['path'],
        index_columns=['date'],
    )
    total_rows = sum(datetime_sample['sizes'])
    rows_per_partition = 24
    samples_per_partition = 24
    index_columns = ('hour',)
    base_seed = 10
    seed_increment = 10
    parallel = False

    divisions = get_index_divisions(
        ds,
        total_rows,
        rows_per_partition,
        samples_per_partition,
        index_columns,
        base_seed,
        seed_increment,
        parallel,
    )
    expected_divisions = [(timedelta(hours=h),) for h in range(0, 24, 6)]
    assert divisions == expected_divisions


