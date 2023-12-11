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
    t = pl.datetime_range(t0, t1, dt, closed='left', eager=True).alias('t')
    day = t.dt.truncate('1d').cast(pl.Date).alias('date')
    hour = (t - day).alias('hour')
    a = pl.Series('a', np.arange(len(t)))
    df = pl.DataFrame([t, day, hour, a])

    null_df = pl.DataFrame([
        pl.Series('t', [None, None], pl.Datetime),
        pl.Series('date', [date(2022, 1, 1), None], pl.Date),
        pl.Series('hour', [None, timedelta(hours=0)], pl.Duration),
        pl.Series('a', [-2, -1], pl.Int64),
    ])
    df = pl.concat([null_df, df])

    divisions = list(range(2, len(df), 24))
    divisions[0] = 0
    divisions.append(len(df))

    sample_path = 'tests/data/datetime_sample.parquet'
    clear_directory(sample_path)
    for i, (start, end) in enumerate(zip(divisions[:-1], divisions[1:])):
        part = df[start:end, :]
        part.write_parquet(os.path.join(sample_path, f'part{2*i}.parquet'))
        part[:0, :].write_parquet(
            os.path.join(sample_path, f'part{2*i+1}.parquet'))

    sizes = (26, 24, 24, 24)
    lower_bounds = (
        (None, timedelta(hours=0), None),
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
        'divisions': divisions,
    }


@pytest.fixture
def date_sample():
    t0 = datetime(2022, 1, 2)
    t1 = datetime(2022, 1, 6)
    dt = timedelta(days=1)
    t = pl.datetime_range(t0, t1, dt, closed='left', eager=True).alias('t')
    day = t.dt.truncate('1d').cast(pl.Date).alias('date')
    x = pl.Series('x', np.arange(len(day)))
    df = pl.DataFrame([day, x])


    divisions = [0, 2, 4]

    sample_path = 'tests/data/date_sample.parquet'
    clear_directory(sample_path)
    for i, (start, end) in enumerate(zip(divisions[:-1], divisions[1:])):
        part = df[start:end, :]
        part.write_parquet(os.path.join(sample_path, f'part{i}.parquet'))

    sizes = (2, 2)
    lower_bounds = (
        (date(2022, 1, 2),),
        (date(2022, 1, 4),),
    )
    upper_bounds = (
        (date(2022, 1, 3),),
        (date(2022, 1, 5),),
    )

    yield {
        'path': sample_path,
        'index_columns': ('date',),
        'sizes': sizes,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'data': df,
        'divisions': divisions,
    }


@pytest.fixture
def output_dir():
    output_path = 'tests/data/result.parquet'
    remove_directory(output_path)
    yield output_path
