import polars as pl


def dataframe_from_schema(schema):
    return pl.DataFrame([pl.Series(c, [], dtype=t) for c, t in schema.items()])


def dataframe_eq(df1, df2):
    return all((df1 == df2).select(pl.col('*').all()).row(0))
