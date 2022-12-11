import padawan as pw
import polars as pl
from datetime import datetime, timedelta
import numpy as np
import os
import shutil
import time
import multiprocessing



def clear_directory(path):
    try:
        shutil.rmtree(path)
    except NotADirectoryError:
        os.remove(path)
    except FileNotFoundError:
        pass
    os.mkdir(path)


def generate_data():
    t0 = datetime(2022, 1, 1)
    t1 = datetime(2022, 1, 5)
    dt = timedelta(hours=1)
    t = pl.date_range(t0, t1, dt, name='t', closed='left')
    day = t.dt.truncate('1d').cast(pl.Date).alias('day')
    hour = (t - day).alias('hour')
    a = pl.Series('a', np.arange(len(t)))
    df = pl.DataFrame([t, day, hour, a])

    clear_directory('data/sample.parquet')
    for i in range(0, len(df), 24):
        part = df[i:i+24, :]
        part.write_parquet(f'data/sample.parquet/part{i}.parquet')


if __name__ == '__main__':
    generate_data()

    ds = pw.read_parquet(
        'data/sample.parquet',
        index_columns=['day', 'hour'],
        parallel=2,
    )
    ds.write_parquet('data/sample_copy.parquet')
    ds = pw.read_parquet('data/sample_copy.parquet')
    print(ds[0].collect())
