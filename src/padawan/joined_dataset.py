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

        schema = None
        if left.known_schema and right.known_schema:
            index_columns = left.index_columns
            schema = left.schema
            for c, t in right.schema.items():
                if c in index_columns:
                    continue
                if c in schema:
                    raise ValueError(f'Duplicate column {repr(c)} in join.')
                schema[c] = t

        super().__init__(
            npartitions=len(divisions) + 1,
            index_columns=left.index_columns,
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
            schema=schema,
        )
        self._divisions = [None] + divisions + [None]
        self._how = how

    def _get_partition(self, partition_index):
        lb = self._divisions[partition_index]
        ub = self._divisions[partition_index + 1]

        left_slice = self._left.slice(lb, ub, inclusive='lower').collect()
        right_slice = self._right.slice(lb, ub, inclusive='lower').collect()
        return left_slice.join(
            right_slice, on=self._index_columns, how=self._how).lazy()


def _join(self, other, how='inner'):
    """Join with another dataset.

    Args:
      other (padawan.Dataset): The dataset to join. `self` and `other` must
        have the same index columns and the join is done on those columns.
        You can use :py:meth:`padawan.Dataset.reindex` and
        :py:meth:`padawan.Dataset.rename` to give both datasets index columns
        with the same name.
      how (str, optional): The type of join to perform. Supported values are
        ``'inner'``, ``'left'`` and ``'outer'``. Defaults to ``'inner'``.

    Returns:
      padawan.Dataset: The joined dataset.

    """
    return JoinedDataset(self, other, how=how)
Dataset.join = _join

