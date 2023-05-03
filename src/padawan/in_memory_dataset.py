import glob
import os
import polars as pl

from .dataset import Dataset, lex_min, lex_max


class InMemoryDataset(Dataset):
    def __init__(self, data, index_columns=()):
        self._data = data.lazy().collect()
        index_columns = tuple(index_columns)
        index = self._data.select(list(index_columns))

        sizes = [len(self._data)]
        if index_columns:
            index = self._data.select(list(index_columns))
            lower_bounds = [lex_min(index)]
            upper_bounds = [lex_max(index)]
        else:
            lower_bounds = [()]
            upper_bounds = [()]
        schema = self._data.schema

        super().__init__(
            1,
            index_columns,
            sizes,
            lower_bounds,
            upper_bounds,
            schema,
        )

    def _get_partition(self, partition_index):
        return self._data.lazy()


def from_polars(data, index_columns=()):
    """Create a single-partition dataset from a polars DataFrame.

    Args:
      data (polars.DataFrame or polars.LazyFrame): The polars DataFrame object
        to use as the (single) partition. If this is a ``polars.LazyFrame``
        object it will be collected.
      index_columns (tuple of str, optional): The columns to use as index.
        Defaults to the empty tuple, i.e. to not designating any columns as
        index.

    Returns:
      padawan.Dataset: A single-partiton dataset containing `data`.

    """
    return InMemoryDataset(data, index_columns=index_columns)
