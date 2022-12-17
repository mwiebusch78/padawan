import pytest
import polars as pl
import numpy as np
import padawan
from datetime import datetime, timedelta
import os
import shutil


def remove_directory(path):
    try:
        shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)
    except FileNotFoundError:
        pass


def clear_directory(path):
    remove_directory(path)
    os.mkdir(path)


@pytest.fixture
def input_dir():
    t0 = datetime(2022, 1, 1)
    t1 = datetime(2022, 1, 5)
    dt = timedelta(hours=1)
    t = pl.date_range(t0, t1, dt, name='t', closed='left')
    day = t.dt.truncate('1d').cast(pl.Date).alias('day')
    hour = (t - day).alias('hour')
    a = pl.Series('a', np.arange(len(t)))
    df = pl.DataFrame([t, day, hour, a])

    partition_size = 24
    sample_path = 'data/sample.parquet'
    clear_directory(sample_path)
    for i in range(len(df)//partition_size):
        part = df[i*partition_size:(i+1)*partition_size, :]
        part.write_parquet(f'data/sample.parquet/part{i}.parquet')

    yield sample_path


@pytest.fixture
def output_dir():
    output_path = 'data/result.parquet'
    remove_directory(output_path)
    yield output_path


def test__scan_parquet__with_index_columns(input_dir):
    ds = padawan.scan_parquet(
        input_dir,
        index_columns=['day', 'hour'],
    )
    assert ds.index_columns == ('day', 'hour')
    assert ds.known_sizes is False
    assert ds.sizes is None
    assert ds.known_bounds is False
    assert ds.lower_bounds is None
    assert ds.upper_bounds is None

