import pytest
import padawan
import math
from datetime import datetime, timedelta
import polars as pl

from padawan.repartitioned_dataset import (
    get_row_divisions,
    get_index_divisions,
)

from fixtures import (
    datetime_sample,
    output_dir,
)

from utils import dataframe_eq


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
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['date'], collect_stats=False)
    )
    rows_per_partition = 24 
    sample_fraction = 1.0
    index_columns = ('hour',)
    base_seed = 10
    seed_increment = 10
    parallel = False
    progress = False

    divisions, sizes, lower_bounds, upper_bounds = get_index_divisions(
        ds,
        rows_per_partition,
        sample_fraction,
        index_columns,
        base_seed,
        seed_increment,
        parallel,
        progress,
    )
    expected_divisions = [(timedelta(hours=h),) for h in range(5, 24, 6)]
    expected_sizes = [22, 24, 24, 24, 4]
    expected_lower_bounds = [(None,)] + expected_divisions
    expected_upper_bounds = [(timedelta(hours=h),) for h in range(4, 24, 6)] \
        + [(timedelta(hours=23),)]
    assert divisions == expected_divisions
    assert sizes == expected_sizes
    assert lower_bounds == expected_lower_bounds
    assert upper_bounds == expected_upper_bounds


def test__repartition__indexed(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex(['hour'])
        .repartition(24)
    )

    assert len(ds) == 5
    assert ds.is_disjoint()

    assert dataframe_eq(
        ds[0].collect(),
        datetime_sample['data'].filter(
            pl.col('hour').is_null()
            | (pl.col('hour') <= timedelta(hours=4))
        ),
    )
    assert dataframe_eq(
        ds[1].collect(),
        datetime_sample['data'].filter(
            (pl.col('hour') >= timedelta(hours=5))
            & (pl.col('hour') <= timedelta(hours=10))
        ),
    )
    assert dataframe_eq(
        ds[2].collect(),
        datetime_sample['data'].filter(
            (pl.col('hour') >= timedelta(hours=11))
            & (pl.col('hour') <= timedelta(hours=16))
        ),
    )
    assert dataframe_eq(
        ds[3].collect(),
        datetime_sample['data'].filter(
            (pl.col('hour') >= timedelta(hours=17))
            & (pl.col('hour') <= timedelta(hours=22))
        ),
    )
    assert dataframe_eq(
        ds[4].collect(),
        datetime_sample['data'].filter(
            pl.col('hour') == timedelta(hours=23)
        ),
    )
    

def test__repartition__no_index(datetime_sample):
    ds = (
        padawan.scan_parquet(datetime_sample['path'])
        .reindex()
        .repartition(10)
    )

    num_partitions = math.ceil(len(datetime_sample['data'])/10)
    assert len(ds) == num_partitions
    assert ds.is_disjoint()
    for i in range(num_partitions):
        assert dataframe_eq(
            ds[i].collect(),
            datetime_sample['data'][10*i:10*(i+1), :],
        )


