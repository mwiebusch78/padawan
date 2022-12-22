import polars as pl

from .dataset import Dataset


def get_row_divisions(partition_sizes, rows_per_partition):
        partition_indices = [0]
        row_indices = [0]
        src_partition = 0
        src_row_index = 0
        dest_row_index = 0
        num_src_partitions = len(partition_sizes)
        while (src_partition, src_row_index) < (num_src_partitions, 0):
            rem_src_rows = partition_sizes[src_partition] - src_row_index
            rem_dest_rows = rows_per_partition - dest_row_index
            if rem_src_rows < rem_dest_rows:
                # add remainder of source partition
                dest_row_index += rem_src_rows
                src_partition += 1
                src_row_index = 0
            elif rem_src_rows == rem_dest_rows:
                # add remainder of source partition and create split
                dest_row_index += rem_src_rows
                src_partition += 1
                src_row_index = 0
                dest_row_index = 0
                if src_partition < num_src_partitions:
                    partition_indices.append(src_partition)
                    row_indices.append(src_row_index)
            else:
                # add part of source partition and create split
                src_row_index += rem_dest_rows
                dest_row_index = 0
                if src_partition < num_src_partitions:
                    partition_indices.append(src_partition)
                    row_indices.append(src_row_index)

        return list(zip(partition_indices, row_indices))


def _sample_partition(part, seed, index_columns, frac):
    return (
        part
        .select(index_columns)
        .collect()
        .sample(frac=frac, seed=seed)
    )


def get_index_divisions(
        ds,
        total_rows,
        rows_per_partition,
        samples_per_partition,
        index_columns,
        base_seed,
        seed_increment,
        parallel,
):
    num_partitions = len(ds)
    sample_frac = max(samples_per_partition/rows_per_partition, 1.0)
    samples_per_partition = int(rows_per_partition*sample_frac)

    extra_args = [
        (base_seed + i*seed_increment,) for i in range(num_partitions)]
    shared_args = (index_columns, sample_frac)
    sample = (
        ds
        .map(
            _sample_partition,
            extra_args=extra_args,
            shared_args=shared_args,
        )
        .collect(parallel=parallel)
        .sort(index_columns)
    )

    divisions = list(sample[::samples_per_partition, :].rows())
    return divisions


class RepartitionedDataset(Dataset):
    def __init__(
            self,
            other,
            rows_per_partition,
            by=None,
            samples_per_partition=100,
            parallel=False,
            base_seed=10,
            seed_increment=10,
    ):
        if not isinstance(other, Dataset):
            raise ValueError('other must be a Dataset object')
        self._other = other.collect_stats(parallel)

        if by is None:
            by = self._other.index_columns

        if not by:
            self._divisions = get_row_divisions(
                self._other.sizes, rows_per_partition)

