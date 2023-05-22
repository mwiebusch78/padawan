import polars as pl

from .dataset import Dataset


def schema_eq(schema1, schema2):
    for (c1, t1), (c2, t2) in zip(schema1.items(), schema2.items()):
        if c1 != c2 or t1 != t2:
            return False
    return True


class ConcatenatedDataset(Dataset):
    def __init__(self, others):
        if not hasattr(others, '__iter__'):
            raise ValueError('`others` argument must be iterable')
        if not all(isinstance(d, Dataset) for d in others):
            raise ValueError('all elements of `others` must be an instances of padawan.Dataset')

        others = [d for d in others if len(d) > 0]
        if not others:
            super().__init__(
                npartitions=0,
                index_columns=(),
                sizes=(),
                lower_bounds=(),
                upper_bounds=(),
                schema={},
            )
            self._others = []
            self._dataset_indices = []
            self._offsets = []
            return

        index_columns = others[0].index_columns
        if not all(d.index_columns == index_columns for d in others):
            raise ValueError('all concatenated datasets must have the same index columns')

        sizes = []
        for d in others:
            if d.sizes is None:
                sizes = None
                break
            sizes.extend(d.sizes)

        lower_bounds = []
        for d in others:
            if d.lower_bounds is None:
                lower_bounds = None
                break
            lower_bounds.extend(d.lower_bounds)

        upper_bounds = []
        for d in others:
            if d.upper_bounds is None:
                upper_bounds = None
                break
            upper_bounds.extend(d.upper_bounds)

        if any(d.schema is None for d in others):
            schema = None
        else:
            schema = others[0].schema
            if not all(schema_eq(d.schema, schema) for d in others):
                raise ValueError('all concatenated datasets must have the same schema')

        dataset_indices = []
        offsets = [0]
        for i, d in enumerate(others):
            dataset_indices.extend([i]*len(d))
            offsets.append(offsets[-1] + len(d))
        del offsets[-1]

        super().__init__(
            npartitions=sum(len(d) for d in others),
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=schema,
        )
        self._others = others
        self._dataset_indices = dataset_indices
        self._offsets = offsets

    def _get_partition(self, partition_index):
        dataset_index = self._dataset_indices[partition_index]
        ds = self._others[dataset_index]
        offset = self._offsets[dataset_index]
        return ds[partition_index - offset]


def concat(datasets):
    """Concatenate multiple datasets.

    Args:
      datasets (list of padawan.Dataset): The datasets to concatenate. All
        datasets must have the same index columns and the same schema.

    Returns:
      padawan.Dataset: The concatenated dataset.

    """
    return ConcatenatedDataset(datasets)

