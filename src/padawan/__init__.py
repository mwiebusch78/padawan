__all__ = [
    'scan_parquet',
    'from_polars',
    'concat',
    'write_metadata',
    'combine',
]

from .version import __version__

from . import dataset

# Load submodules in correct order.
# Each of these adds methods to dataset.Dataset
from . import reindexed_dataset
from . import mapped_dataset
from . import renamed_dataset
from . import persisted_dataset
from . import collated_dataset
from . import sliced_dataset
from . import repartitioned_dataset
from . import combined_dataset

from .persisted_dataset import scan_parquet
from .in_memory_dataset import from_polars
from .dataset import Dataset
from .concatenated_dataset import concat
from .write_metadata import write_metadata
from .combined_dataset import combine
