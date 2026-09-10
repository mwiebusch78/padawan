import polars as pl

from .dataset import Dataset, dataframe_from_schema
from .ordering import lex_key


class CombinedDataset(Dataset):
    def __init__(self, datasets, func):
        if not hasattr(datasets, '__iter__'):
            raise ValueError('`datasets` must be iterable.')
        datasets = list(datasets)
        if not datasets:
            raise ValueError('`datasets` cannot be empty.')
        index_columns = tuple(datasets[0].index_columns)
        for ds in datasets:
            if not isinstance(ds, Dataset):
                raise ValueError(
                    'Elements of `datasets` must be instances of '
                    'padawan.Dataset'
                )
            if not ds.known_bounds:
                raise ValueError(
                    'Bounds of elements of `datasets` dataset must be known. '
                    'Try using reindex first.'
                )
            if not ds.index_columns == index_columns:
                raise ValueError(
                    'All datasets in `datasets` must have the same '
                    'index columns.'
                )
        if not hasattr(func, '__call__'):
            raise ValueError('`func` must be callable.')

        self._datasets = datasets

        divisions = sum((ds.lower_bounds for ds in datasets), ())
        divisions = sorted(set(divisions), key=lex_key)

        super().__init__(
            npartitions=len(divisions) + 1,
            index_columns=index_columns,
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
            schema=None,
        )
        self._divisions = [None] + divisions + [None]
        self._func = func

    def _get_partition(self, partition_index):
        lb = self._divisions[partition_index]
        ub = self._divisions[partition_index + 1]

        slices = [
            ds.slice(lb, ub, inclusive='lower').collect()
            for ds in self._datasets
        ]
        return self._func(*slices).lazy()


def combine(datasets, func):
    """Combine multiple datasets using a custom function.

    Args:
      datasets (list of padawan.Dataset): The datasets to combine. All datasets
        must have the same index columns.
      func (callable): The function used to combine partitions of the datasets.
        Each partition in the output dataset is obtained by calling `func`
        on slices of the datasets in `datasets` where the index columns cover
        the same range.

    Returns:
      padawan.Dataset: The combined dataset.

    """
    return CombinedDataset(datasets, func)

