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
            schema=None,
            preserves='none',
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be an instance of padawan.Dataset')
        self._other = other
        self._func = func

        preserves_sizes = preserves in ['all', 'sizes']
        preserves_bounds = preserves in ['all', 'bounds']
        if preserves_bounds and index_columns is not None:
            if tuple(index_columns) != other.index_columns:
                raise ValueError(
                    'Index columns cannot change when bounds are preserved.')
        
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
            schema=schema,
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
        schema=None,
        preserves='none',
):
    """Apply a function to all partitions.

    Args:
      func (callable): The function to apply. It should return a
        ``polars.DataFrame`` or ``polars.LazyFrame`` and support the following
        signature::

          func(part,
               extra_arg_1, extra_arg_2, ...,
               shared_arg_1, shared_arg_2, ...)

        where `part` is a ``polars.LazyFrame`` with the partition data,
        `extra_arg_1` etc. are partition-specific arguments specified via
        `extra_args` (see below) and `shared_arg_1` etc are shared arguments
        specified via `shared_args` (see below).
      extra_args (list of tuples, optional): Extra partition-specific arguments
        passed to func. The length of the list must equal the number of
        partitions and each tuple in the list is unpacked and then passed
        as additional arguments to the function call for the corresponding
        partition. Defaults to ``None``, in which case no extra arguments are
        passed.
      shared_args (tuple, optional): This tuple is unpacked and passed as
        extra arguments to *every* call of `func`. Defaults to ``None``, in
        which case no shared arguments are passed.
      index_columns (tuple of str, optional): The new index columns to use
        after `func` is applied. (Note that `func` may change the schema of the
        dataset, so the old index columns might not exist anymore after `func`
        is applied.) Defaults to ``None``, in which case the old index columns
        are used.
      schema (dict, optional): The schema of the dataset after the map. Defaults
        to ``None``, in which case the schema will be unknown.
      preserves (str, optional): Specifies which part of the metadata is
        preserved by `func`. Possible values are:

          ``'none'``
            No metadata is preserved.
          ``'sizes'``
            Partition sizes (number of rows) are preserved.
          ``'bounds'``
            Partition bounds (``self.lower_bounds`` and
            ``self.upper_bounds``) are preserved.
          ``'all'``
            Both partition sizes and bounds are preserved.

        Defaults to ``'none'``.
        
        Note that the behaviour of `func` is not checked. If you specify
        ``preserves='bounds'`` but your `func` actually changes the bounds
        this will lead to incorrect behaviour downstream.

    Returns:
      padawan.Dataset: A dataset with `func` applied to each partition.
    """
    return MappedDataset(
        self,
        func,
        extra_args=extra_args,
        shared_args=shared_args,
        index_columns=index_columns,
        schema=schema,
        preserves=preserves,
    )
Dataset.map = _map

