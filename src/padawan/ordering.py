import polars as pl
import functools


def _null_lt(col, val):
    if val is None:
#         return pl.lit(False)
        # Work around polars bug with broadcasting boolean literals
        return pl.col(col).is_null() & pl.col(col).is_not_null()
    return pl.col(col).is_null() | (pl.col(col) < val)


def _null_leq(col, val):
    if val is None:
        return pl.col(col).is_null()
    return pl.col(col).is_null() | (pl.col(col) <= val)


def _null_gt(col, val):
    if val is None:
        return ~pl.col(col).is_null()
    return pl.col(col) > val


def _null_geq(col, val):
    if val is None:
#         return pl.lit(True)
        # Work around polars bug with broadcasting boolean literals
        return pl.col(col).is_null() | pl.col(col).is_not_null()
    return pl.col(col) >= val


def columns_lt(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_lt(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_lt(c, b) | ((pl.col(c) == b) & columns_lt(c_rem, b_rem))


def columns_leq(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_leq(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_lt(c, b) | ((pl.col(c) == b) & columns_leq(c_rem, b_rem))


def columns_gt(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_gt(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_gt(c, b) | ((pl.col(c) == b) & columns_gt(c_rem, b_rem))


def columns_geq(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_geq(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_gt(c, b) | ((pl.col(c) == b) & columns_geq(c_rem, b_rem))


def nullable_cmp(a, b):
    if a is None:
        if b is None:
            return 0
        return -1
    if b is None:
        return 1
    if a == b:
        return 0
    elif a < b:
        return -1
    return 1

nullable_key = functools.cmp_to_key(nullable_cmp)


def lex_cmp(a, b):
    hcmp = nullable_cmp(a[0], b[0])
    if len(a) == 1 or hcmp != 0:
        return hcmp
    return lex_cmp(a[1:], b[1:])

lex_key = functools.cmp_to_key(lex_cmp)
