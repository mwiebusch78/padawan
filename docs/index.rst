.. padawan documentation master file, created by
   sphinx-quickstart on Fri Jan 20 19:33:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to padawan's documentation!
===================================

`padawan`_ is a tool for out-of-core processing of partitioned tabular datasets
which are too large to hold completely in memory. It uses `polars`_ for 
representing and manipulating tabular data in memory and the `parquet`_ format
for storing partitions on disk.

`polars`_ is a library for 'SQL-type' in-memory data manipulation. It provides
roughly the same functionality as `pandas`_ but has a cleaner API and superiour
performance -- especially on multi-core computers since it consistently
utilises all available cores. While `polars`_ has some capabilities for
out-of-core processing these are (currently) somewhat limited. For example,
`polars`_ cannot handle situations where the result of a computation is too
large to fit in memory. This is where `padawan`_ can help.

The central object in `padawan`_ is the :py:class:`padawan.Dataset`. It has the
semantics of a list of `polars`_ dataframes (``polars.LazyFrame`` objects, to
be exact) which represent the partitions. Each dataset specifies a set of
*index columns* and keeps track of the upper and lower bounds of the index
columns for each partition. This means that certain operations like slicing or
joins on the index columns can be carried out efficiently and without visiting
the full set of partitions. Furthermore, the supported operations are carried
out in a lazy fashion and partitions are only pulled into memory when needed.

Contrary to packages like `pyspark`_ or `dask.dataframe`_ `padawan`_ does not
attempt to implement its own, complete dataframe API. It focuses on
functionality for managing the partitioning (collate, repartition etc.) and on
operations which can be done efficiently on partitioned data with known
partition boundaries (slicing, joins). All other forms of data manipulation are
left to the `polars`_ API which can be accessed directly  by mapping a custom
function over the partitions via :py:meth:`padawan.Dataset.map`.

Since `polars`_ makes efficient use of all available CPUs in most situations,
parallelisation can usually be left to `polars`_. However, for cases where a
substantial part of the computation is done by the Python interpreter (and is
therefore subject to the limitations of the GIL) `padawan`_ also offers a
convenient mechanism for parallelising computations via the ``multiprocessing``
module. Furthermore, it uses `cloudpickle`_ to allow the parallelisation of
lambda functions. Note that, unlike `pyspark`_ or `dask.dataframe`_ `padawan`_
is only intended for computations on a single node and does not offer
functionality for distributing computations on a cluster. 

.. _padawan: https://github.com/mwiebusch78/padawan
.. _polars: https://pola-rs.github.io/polars/py-polars/html/index.html
.. _parquet: https://parquet.apache.org/
.. _pandas: https://pandas.pydata.org/
.. _dask.dataframe: https://docs.dask.org/en/stable/dataframe.html
.. _pyspark: https://spark.apache.org/docs/latest/api/python/
.. _cloudpickle: https://pypi.org/project/cloudpickle/


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
