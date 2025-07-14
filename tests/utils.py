import polars as pl


def dataframe_from_schema(schema):
    return pl.DataFrame([pl.Series(c, [], dtype=t) for c, t in schema.items()])


def dataframe_eq(df1, df2):
    return all((df1 == df2).select(pl.col('*').all()).row(0))


def check_bounds_and_sizes(ds):
    assert ds.known_bounds
    assert ds.known_sizes
    index_columns = ds.index_columns
    for part, lb, ub, size in zip(
            ds, ds.lower_bounds, ds.upper_bounds, ds.sizes):
        part = part.sort(index_columns).collect()
        assert len(part) == size
        if len(part) > 0 and ds.index_columns:
            first_index = part.select(index_columns).row(0)
            last_index = part.select(index_columns).row(-1)
            assert first_index == lb
            assert last_index == ub
