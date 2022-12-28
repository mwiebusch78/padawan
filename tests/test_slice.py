import pytest
import padawan
from datetime import date, datetime, timedelta


from fixtures import datetime_sample


# def test__slice__unknown_stats(datetime_sample):
#     ds = padawan.scan_parquet(
#         datetime_sample['path'],
#         index_columns=['date', 'hour'],
#     )
#     lb = (date(2022, 1, 2), timedelta(hours=7))
#     ub = (date(2022, 1, 3), timedelta(hours=18))
#     partitions = ds._slice(lb, ub)
#     assert partitions is None
# 
# 
# def test__slice__known_stats(datetime_sample):
#     ds = (
#         padawan.scan_parquet(
#             datetime_sample['path'],
#             index_columns=['date', 'hour'],
#         )
#         .collect_stats()
#     )
#     lb = (date(2022, 1, 2), timedelta(hours=7))
#     ub = (date(2022, 1, 3), timedelta(hours=18))
#     partitions = ds._slice(lb, ub)
#     assert partitions == [1, 2]
# 
# 
# def test__slice__exclusive_upper(datetime_sample):
#     ds = (
#         padawan.scan_parquet(
#             datetime_sample['path'],
#             index_columns=['date', 'hour'],
#         )
#         .collect_stats()
#     )
#     lb = (date(2022, 1, 2), timedelta(hours=0))
#     ub = (date(2022, 1, 4), timedelta(hours=0))
#     partitions = ds._slice(lb, ub)
#     assert partitions == [1, 2]
# 
# 
# def test__slice__column_subset(datetime_sample):
#     ds = (
#         padawan.scan_parquet(
#             datetime_sample['path'],
#             index_columns=['date', 'hour'],
#         )
#         .collect_stats()
#     )
#     lb = (date(2022, 1, 2),)
#     ub = (date(2022, 1, 4),)
#     partitions = ds._slice(lb, ub, index_columns=['date'])
#     assert partitions == [1, 2]
# 
# 
# def test__slice__column_superset(datetime_sample):
#     ds = (
#         padawan.scan_parquet(
#             datetime_sample['path'],
#             index_columns=['date', 'hour'],
#         )
#         .collect_stats()
#     )
#     lb = (date(2022, 1, 2), timedelta(hours=0), datetime(2022, 1, 2))
#     ub = (date(2022, 1, 4), timedelta(hours=0), datetime(2022, 1, 4))
#     with pytest.raises(ValueError):
#         partitions = ds._slice(lb, ub, index_columns=['date', 'hour', 't'])
# 
