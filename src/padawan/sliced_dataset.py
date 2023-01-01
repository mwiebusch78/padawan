import polars as pl
import functools

from .dataset import Dataset
from .ordering import lex_key, columns_geq, columns_lt, _null_lt


class SlicedDataset(Dataset):
    def __init__(self, other, lb=None, ub=None, index_columns=None):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        if not other.known_bounds:
            raise ValueError(
                'Bounds must be known for slicing. Use collect_stats first.')
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

        # Project bounds.
        other_lbs = [b[:len(index_columns)] for b in self._other.lower_bounds]
        other_ubs = [b[:len(index_columns)] for b in self._other.upper_bounds]

        # Determine overlapping partitions.
        partitions = list(range(len(self._other)))
        if lb is not None:
            lb_key = lex_key(lb)
            partitions = [
                p for p in partitions if lb_key <= lex_key(other_ubs[p])]
        if ub is not None:
            ub_key = lex_key(ub)
            partitions = [
                p for p in partitions if lex_key(other_lbs[p]) < ub_key]

        # Get bounds for selected partitions.
        lower_bounds = [other_lbs[p] for p in partitions]
        upper_bounds = [other_ubs[p] for p in partitions]

        # Determine sizes for new dataset.

        sizes = None

        lb_irrelevant = False
        if lb is None:
            lb_irrelevant = True
        else:
            lb_key = lex_key(lb)
            lb_irrelevant = all(lb_key >= lex_key(b) for b in lower_bounds)

        ub_irrelevant = False
        if ub is None:
            ub_irrelevant = True
        else:
            ub_key = lex_key(ub)
            ub_irrelevant = all(lex_key(b) < ub_key for b in upper_bounds)

        if lb_irrelevant and ub_irrelevant:
            other_sizes = self._other.sizes
            if other_sizes is not None:
                sizes = [other_sizes[p] for p in partitions]

        # Determine bounds for new dataset.

        if lb is not None:
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
        other_part_index = self._partitions[partition_index]
        part = self._other[other_part_index]
        part_lb = self._other.lower_bounds[other_part_index]
        part_lb = part_lb[:len(self._index_columns)]
        part_ub = self._other.upper_bounds[other_part_index]
        part_ub = part_ub[:len(self._index_columns)]
        if self._lb is not None and lex_key(part_lb) < lex_key(self._lb):
            part = part.filter(columns_geq(self._index_columns, self._lb))
        if self._ub is not None and lex_key(self._ub) <= lex_key(part_ub):
            part = part.filter(columns_lt(self._index_columns, self._ub))

        return part


def _slice(self, lb=None, ub=None, index_columns=None):
    return SlicedDataset(self, lb=lb, ub=ub, index_columns=index_columns)

Dataset.slice = _slice