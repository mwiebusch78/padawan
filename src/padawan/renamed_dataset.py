import polars as pl

from .dataset import Dataset


class RenamedDataset(Dataset):
    def __init__(
            self,
            other,
            mapping,
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be an instance of padawan.Dataset')
        self._other = other
        self._mapping = mapping.copy()

        index_columns = tuple(
            self._mapping.get(c, c) for c in self._other.index_columns)

        schema = None
        if self._other.schema is not None:
            schema = dict(
                (self._mapping.get(c, c), t)
                for c, t in self._other.schema.items()
            )

        super().__init__(
            npartitions=len(other),
            index_columns=index_columns,
            sizes=self._other.sizes,
            lower_bounds=self._other.lower_bounds,
            upper_bounds=self._other.upper_bounds,
            schema=schema,
        )

    def _get_partition(self, partition_index):
        return self._other[partition_index].rename(self._mapping)


def _rename(self, mapping):
    """Rename columns of the dataset.

    Args:
      mapping (dict): A dictionary mapping old column names to the new ones.

    Returns:
      padawan.Dataset: The dataset with renamed columns.

    """
    return RenamedDataset(self, mapping)
Dataset.rename = _rename

