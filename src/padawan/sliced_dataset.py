import polars as pl
import functools

from .dataset import Dataset


def _null_lt(col, val):
    if val is None:
        return False
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
        return True
    return pl.col(col) >= val


def _columns_lt(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_lt(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_lt(c, b) | ((pl.col(c) == b) & _columns_lt(c_rem, b_rem))


def _columns_leq(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_leq(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_lt(c, b) | ((pl.col(c) == b) & _columns_leq(c_rem, b_rem))


def _columns_gt(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_gt(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_gt(c, b) | ((pl.col(c) == b) & _columns_gt(c_rem, b_rem))


def _columns_geq(columns, bound):
    c = columns[0]
    b = bound[0]
    if len(columns) == 1:
        return _null_geq(c, b)
    c_rem = columns[1:]
    b_rem = bound[1:]
    return _null_gt(c, b) | ((pl.col(c) == b) & _columns_geq(c_rem, b_rem))


def nullable_cmp(a, b):
    if a == b:
        return 0
    elif a is None or a < b:
        return -1
    else:
        return 1

nullable_key = functools.cmp_to_key(nullable_cmp)


def lex_cmp(a, b):
    hcmp = nullable_cmp(a[0], b[0])
    if len(a) == 1 or hcmp != 0:
        return hcmp
    return lex_cmp(a[1:], b[1:])

lex_key = functools.cmp_to_key(lex_cmp)


class SlicedDataset(Dataset):
    def __init__(self, other, lb=None, ub=None, index_columns=None):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        if lb is not None:
            lb = tuple(lb)
        if ub is not None:
            ub = tuple(ub)
        self._other = other

        # Determine index columns.
        if index_columns is None:
            index_columns = self._other.index_columns
        else:
            index_columns = tuple(index_columns)
        if len(index_columns) > len(self._other.index_columns) \
                or index_columns != \
                self._other.index_columns[:len(index_columns)]:
            raise ValueError(
                'index_columns must be a subset of other.index_columns')

        # Project bounds if necessary.
        other_lbs = self._other.lower_bounds
        other_ubs = self._other.upper_bounds
        if other_lbs is not None:
            other_lbs = [b[:len(index_columns)] for b in other_lbs]
        if other_ubs is not None:
            other_ubs = [b[:len(index_columns)] for b in other_ubs]

        # Determine overlapping partitions.
        partitions = list(range(ds._npartitions))
        if lb is not None and other_ubs is not None:
            lb_key = lex_key(lb)
            partitions = [
                p for p in partitions if lb_key <= lex_key(other_ubs[p])]
        if ub is not None and other_lbs is not None:
            ub_key = lex_key(ub)
            partitions = [
                p for p in partitions if lex_key(other_lbs[p]) < ub_key]

        # Get bounds for selected partitions.
        lower_bounds = None
        upper_bounds = None
        if other_lbs is not None:
            lower_bounds = [other_lbs[p] for p in partitions]
        if other_ubs is not None:
            upper_bounds = [other_ubs[p] for p in partitions]

        # Determine sizes for new dataset.

        sizes = None

        lb_irrelevant = False
        if lb is None:
            lb_irrelevant = True
        elif lower_bounds is not None:
            lb_key = lex_key(lb)
            lb_irrelevant = all(lb_key >= lex_key(b) for b in lower_bounds)

        ub_irrelevant = False
        if ub is None:
            ub_irrelevant = True
        elif upper_bounds is not None:
            ub_key = lex_key(ub)
            ub_irrelevant = all(lex_key(b) < ub_key for b in upper_bounds)

        if lb_irrelevant and ub_irrelevant:
            other_sizes = self._other.sizes
            if other_sizes is not None:
                sizes = [other_sizes[p] for p in partitions]

        # Determine bounds for new dataset.

        if other_lbs is not None and lb is not None:
            lower_bounds = [max(b, lb, key=lex_key) for b in lower_bounds]

        # Initialise dataset.
        super().__init__(
            npartitions=len(partitions),
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self._partitions = partitions
        self._lb = lb
        self._ub = ub

    def __getitem__(self, partition_index):
        part = self._other[self._partitions[partition_index]]
        part_lb = None
        part_ub = None
        if self.known_bounds:
            part_lb = self._lower_bounds[partition_index]
            part_ub = self._upper_bounds[partition_index]
        if self._lb is not None and (
                part_lb is None or lex_key(part_lb) < lex_key(self._lb)):
            part = part.filter(_columns_geq(self._index_columns, self._lb))
        if self._ub is not None and (
                part_ub is None or lex_key(self._ub) <= lex_key(part_ub)):
            part = part.filter(_columns_lt(self._index_columns, self._ub))

        return part

