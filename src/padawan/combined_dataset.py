import polars as pl

from .dataset import Dataset, dataframe_from_schema
from .ordering import lex_key


class CombinedDataset(Dataset):
    def __init__(self, datasets, func, shared_args=None):
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
        if shared_args is None:
            shared_args = ()
        self._shared_args = tuple(shared_args)

        divisions = sum((ds.lower_bounds for ds in datasets), ())
        divisions = sorted(set(divisions), key=lex_key)

        schema = None
        if all(ds.known_schema for ds in datasets):
            parts = [dataframe_from_schema(ds.schema).lazy() for ds in datasets]
            schema = func(*parts, *shared_args).lazy().collect().schema

        super().__init__(
            npartitions=len(divisions),
            index_columns=index_columns,
            sizes=None,
            lower_bounds=None,
            upper_bounds=None,
            schema=schema,
        )
        self._divisions = divisions + [None]
        self._func = func

    def _get_partition(self, partition_index):
        lb = self._divisions[partition_index]
        ub = self._divisions[partition_index + 1]

        slices = [
            ds.slice(lb, ub, inclusive='lower').collect().lazy()
            for ds in self._datasets
        ]
        return self._func(*slices, *self._shared_args).lazy()


def combine(datasets, func, shared_args=None):
    """Combine multiple datasets using a custom function.

    Args:
      datasets (list of padawan.Dataset): The datasets to combine. All datasets
        must have the same index columns.
      func (callable): The function used to combine partitions of the datasets.
        Each partition in the output dataset is obtained by calling `func`
        on slices of the datasets in `datasets` where the index columns cover
        the same range. Note that, to determine the schema of the resulting
        dataset, one call to `func` is made where all slices are empty
        dataframes, and the schema of the returned dataframe is used as the
        output schema.
      shared_args (tuple, optional): List of shared arguments that are passed
        to `func` on every call. The shared arguments are passed as positional
        arguments after the dataset slices.
      schema (dict, optional): The schema of the output dataset. Defaults to
        ``None``, in which case the schema of the resulting dataset will be
        unknown.

    Returns:
      padawan.Dataset: The combined dataset.

    """
    return CombinedDataset(datasets, func, shared_args=shared_args)


def _join_parts(left, right, on, how):
    return left.join(right, on=on, how=how)


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
    return combine(
        datasets=[self, other],
        func=_join_parts,
        shared_args=(self.index_columns, how),
    )
Dataset.join = _join

