import polars as pl
import functools

from .dataset import Dataset
from .ordering import lex_key, columns_geq, columns_gt, columns_leq, columns_lt


class SlicedDataset(Dataset):
    def __init__(self, other, lb=None, ub=None, inclusive='lower'):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        if not other.known_bounds:
            raise ValueError(
                'Bounds must be known for slicing. Use reindex first.')
        if inclusive == 'none':
            lower_inclusive = False
            upper_inclusive = False
        elif inclusive == 'lower':
            lower_inclusive = True
            upper_inclusive = False
        elif inclusive == 'upper':
            lower_inclusive = False
            upper_inclusive = True
        elif inclusive == 'both':
            lower_inclusive = True
            upper_inclusive = True
        else:
            raise ValueError(
                "inclusive must be 'none', 'lower', 'upper' or 'both'")
        if lb is not None:
            lb = tuple(lb)
        if ub is not None:
            ub = tuple(ub)
        self._other = other

        index_columns = self._other.index_columns
        other_lbs = self._other.lower_bounds
        other_ubs = self._other.upper_bounds

        # Determine overlapping partitions.
        partitions = list(range(len(self._other)))
        if lb is not None:
            lbcols = len(lb)
            if lbcols > len(index_columns):
                raise ValueError(
                    'Lower bound must be a tuple of the same length or '
                    'shorter than index_columns'
                )
            lb_key = lex_key(lb)
            if lower_inclusive:
                partitions = [
                    p for p in partitions
                    if lb_key <= lex_key(other_ubs[p][:lbcols])
                ]
            else:
                partitions = [
                    p for p in partitions
                    if lb_key < lex_key(other_ubs[p][:lbcols])
                ]
        if ub is not None:
            ubcols = len(ub)
            if ubcols > len(index_columns):
                raise ValueError(
                    'Upper bound must be a tuple of the same length or '
                    'shorter than index_columns'
                )
            ub_key = lex_key(ub)
            if upper_inclusive:
                partitions = [
                    p for p in partitions
                    if lex_key(other_lbs[p][:ubcols]) <= ub_key
                ]
            else:
                partitions = [
                    p for p in partitions
                    if lex_key(other_lbs[p][:ubcols]) < ub_key
                ]

        # Get bounds for selected partitions.
        lower_bounds = [other_lbs[p] for p in partitions]
        upper_bounds = [other_ubs[p] for p in partitions]

        # Determine sizes for new dataset.

        sizes = None

        lb_irrelevant = False
        if lb is None:
            lb_irrelevant = True
        else:
            if lower_inclusive:
                lb_irrelevant = all(
                    lb_key <= lex_key(b[:lbcols]) for b in lower_bounds)
            else:
                lb_irrelevant = all(
                    lb_key < lex_key(b[:lbcols]) for b in lower_bounds)

        ub_irrelevant = False
        if ub is None:
            ub_irrelevant = True
        else:
            if upper_inclusive:
                ub_irrelevant = all(
                    lex_key(b[:ubcols]) <= ub_key for b in upper_bounds)
            else:
                ub_irrelevant = all(
                    lex_key(b[:ubcols]) < ub_key for b in upper_bounds)

        if lb_irrelevant and ub_irrelevant:
            other_sizes = self._other.sizes
            if other_sizes is not None:
                sizes = [other_sizes[p] for p in partitions]

        # Determine bounds for new dataset.

        if lb is not None:
            if lower_inclusive and lbcols == len(index_columns):
                lower_bounds = [max(b, lb, key=lex_key) for b in lower_bounds]
            if upper_inclusive and ubcols == len(index_columns):
                upper_bounds = [min(b, ub, key=lex_key) for b in upper_bounds]

        # Initialise dataset.
        super().__init__(
            npartitions=len(partitions),
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=self._other._schema,
        )
        self._partitions = partitions
        self._lb = lb
        self._ub = ub
        self._lower_inclusive = lower_inclusive
        self._upper_inclusive = upper_inclusive

    def _get_partition(self, partition_index):
        other_part_index = self._partitions[partition_index]
        part = self._other[other_part_index]
        part_lb = self._other.lower_bounds[other_part_index]
        part_ub = self._other.upper_bounds[other_part_index]
        lb_index_columns = self._index_columns
        ub_index_columns = self._index_columns

        if self._lb is not None:
            ncols = len(self._lb)
            part_lb = part_lb[:ncols]
            lb_index_columns = lb_index_columns[:ncols]
        if self._ub is not None:
            ncols = len(self._ub)
            part_ub = part_ub[:ncols]
            ub_index_columns = ub_index_columns[:ncols]

        if self._lower_inclusive:
            if self._lb is not None and lex_key(part_lb) < lex_key(self._lb):
                part = part.filter(columns_geq(lb_index_columns, self._lb))
        else:
            if self._lb is not None and lex_key(part_lb) <= lex_key(self._lb):
                part = part.filter(columns_gt(lb_index_columns, self._lb))
        if self._upper_inclusive:
            if self._ub is not None and lex_key(self._ub) < lex_key(part_ub):
                part = part.filter(columns_leq(ub_index_columns, self._ub))
        else:
            if self._ub is not None and lex_key(self._ub) <= lex_key(part_ub):
                part = part.filter(columns_lt(ub_index_columns, self._ub))

        return part


def _slice(self, lb=None, ub=None, inclusive='lower'):
    """Take a slice of the dataset using the current index columns.

    Args:
      lb (tuple, optional): A tuple with the same length as or shorter than
        ``self.index_columns`` specifying the lower bound of the
        slice. If `lb` is shorter than ``self.index_columns`` then slicing
        is done on the first ``len(lb)`` index columns only. Defaults to
        ``None``, in which case there is no lower bound.
      ub (tuple, optional): A tuple with the same length as or shorter than
        ``self.index_columns`` specifying the upper bound of the
        slice. If `ub` is shorter than ``self.index_columns`` then slicing
        is done on the first ``len(ub)`` index columns only. Defaults to
        ``None``, in which case there is no upper bound.
      inclusive (str, optional): Specifies which of the bounds are inclusive.
        Allowed values are ``'none'``, ``'lower'``, ``'upper'`` or ``'both'``.
        Defaults to ``'lower'``.

    Returns:
      padawan.Dataset: The slice of the dataset.

    """
    return SlicedDataset(self, lb=lb, ub=ub, inclusive=inclusive)

Dataset.slice = _slice
