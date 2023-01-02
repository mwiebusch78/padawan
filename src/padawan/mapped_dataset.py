import polars as pl

from .dataset import Dataset


class MappedDataset(Dataset):
    def __init__(
            self,
            other,
            func,
            extra_args=None,
            shared_args=None,
            index_columns=None,
            preserves='none',
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be an instance of padawan.Dataset')
        self._other = other
        self._func = func

        preserves_sizes = preserves in ['all', 'sizes']
        preserves_bounds = preserves in ['all', 'bounds']
        
        sizes = None
        lower_bounds = None
        upper_bounds = None
        if index_columns is None:
            index_columns = other.index_columns
        else:
            index_columns = tuple(index_columns)
        if preserves_sizes and other.known_sizes:
            sizes = other.sizes
        if preserves_bounds and other.known_bounds:
            if index_columns == other.index_columns[:len(index_columns)]:
                lower_bounds = [
                    b[:len(index_columns)] for b in other.lower_bounds]
                upper_bounds = [
                    b[:len(index_columns)] for b in other.upper_bounds]
            else:
                raise ValueError(
                    'Index columns must be compatible when bounds are not '
                    'preserved.')

        self._extra_args = extra_args
        self._shared_args = () if shared_args is None else tuple(shared_args)

        super().__init__(
            npartitions=len(other),
            index_columns=index_columns,
            sizes=sizes,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            schema=None,
        )

    def _get_partition(self, partition_index):
        if self._extra_args is None:
            return self._func(
                self._other[partition_index],
                *self._shared_args).lazy()
        return self._func(
            self._other[partition_index],
            *self._extra_args[partition_index],
            *self._shared_args).lazy()


def _map(
        self,
        func,
        extra_args=None,
        shared_args=None,
        index_columns=None,
        preserves='none',
):
    return MappedDataset(
        self,
        func,
        extra_args=extra_args,
        shared_args=shared_args,
        index_columns=index_columns,
        preserves=preserves,
    )
Dataset.map = _map

