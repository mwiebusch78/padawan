import pytest
import polars as pl
import numpy as np
from datetime import datetime, date, timedelta
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
    os.makedirs(path)


@pytest.fixture
def datetime_sample():
    t0 = datetime(2022, 1, 1)
    t1 = datetime(2022, 1, 5)
    dt = timedelta(hours=1)
    t = pl.date_range(t0, t1, dt, name='t', closed='left')
    day = t.dt.truncate('1d').cast(pl.Date).alias('date')
    hour = (t - day).alias('hour')
    a = pl.Series('a', np.arange(len(t)))
    df = pl.DataFrame([t, day, hour, a])

    partition_size = 24
    sample_path = 'tests/data/datetime_sample.parquet'
    clear_directory(sample_path)
    for i in range(len(df)//partition_size):
        part = df[i*partition_size:(i+1)*partition_size, :]
        part.write_parquet(os.path.join(sample_path, f'part{i}.parquet'))


    sizes = (24, 24, 24, 24)
    lower_bounds = (
        (date(2022, 1, 1), timedelta(hours=0), datetime(2022, 1, 1, 0)),
        (date(2022, 1, 2), timedelta(hours=0), datetime(2022, 1, 2, 0)),
        (date(2022, 1, 3), timedelta(hours=0), datetime(2022, 1, 3, 0)),
        (date(2022, 1, 4), timedelta(hours=0), datetime(2022, 1, 4, 0)),
    )
    upper_bounds = (
        (date(2022, 1, 1), timedelta(hours=23), datetime(2022, 1, 1, 23)),
        (date(2022, 1, 2), timedelta(hours=23), datetime(2022, 1, 2, 23)),
        (date(2022, 1, 3), timedelta(hours=23), datetime(2022, 1, 3, 23)),
        (date(2022, 1, 4), timedelta(hours=23), datetime(2022, 1, 4, 23)),
    )

    yield {
        'path': sample_path,
        'index_columns': ('date', 'hour', 't'),
        'sizes': sizes,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'data': df,
    }


@pytest.fixture
def output_dir():
    output_path = 'tests/data/result.parquet'
    remove_directory(output_path)
    yield output_path
