import polars as pl

from .dataset import Dataset, dataframe_from_schema
from .ordering import lex_key


class JoinedDataset(Dataset):
    def __init__(self, left, right, how='inner'):
        if not isinstance(left, Dataset):
            raise ValueError('left must be an instance of padawan.Dataset')
        if not left.known_bounds:
            raise ValueError(
                'Bounds of left dataset must be known. '
                'Try using reindex first.')

        if not isinstance(right, Dataset):
            raise ValueError('right must be an instance of padawan.Dataset')
        if not right.known_bounds:
            raise ValueError(
                'Bounds of right dataset must be known. '
                'Try using reindex first.')
        if not left.index_columns == right.index_columns:
            raise ValueError(
                'Index columns of left and right dataset must be the same.')

        if how not in ['left', 'inner', 'outer']:
            raise ValueError(
                'Only left, inner and outer joins are supported.')

        self._left = left
        self._right = right

        divisions = left.lower_bounds + right.lower_bounds
        divisions = sorted(set(divisions), key=lex_key)

        super().__init__(
            npartitions=len(divisions) + 1,
            index_columns=left.index_columns,
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
            schema=None,
        )
        self._divisions = [None] + divisions + [None]
        self._how = how

    def _get_partition(self, partition_index):
        lb = self._divisions[partition_index]
        ub = self._divisions[partition_index + 1]

        left_slice = self._left.slice(lb, ub).collect()
        right_slice = self._right.slice(lb, ub).collect()
        return left_slice.join(
            right_slice, on=self._index_columns, how=self._how).lazy()


def _join(self, other, how='inner'):
    return JoinedDataset(self, other, how=how)
Dataset.join = _join
