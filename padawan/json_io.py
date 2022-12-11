import json
from datetime import datetime, date, timedelta
import re


_timedelta_regex = re.compile(r'^(-?[\d]+)d([\d]+)s([\d]+)u$')


class DecodingError(Exception):
    pass


def _encode(obj):
    if isinstance(obj, datetime):
        return {'$datetime': obj.isoformat()}
    elif isinstance(obj, date):
        return {'$date': obj.isoformat()}
    elif isinstance(obj, timedelta):
        return {'$timedelta': f'{obj.days}d{obj.seconds}s{obj.microseconds}u'}
    else:
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def _decode(obj):
    if len(obj) == 1:
        key, value = next(iter(obj.items()))
        if key == '$datetime':
            return datetime.fromisoformat(value)
        elif key == '$date':
            return date.fromisoformat(value)
        elif key == '$timedelta':
            match = _timedelta_regex.match(value)
            if not match:
                raise DecodingError(
                    f'Expecting timedelta expression but got {repr(value)}')
            return timedelta(
                days=int(match.group(1)),
                seconds=int(match.group(2)),
                microseconds=int(match.group(3)),
            )
    return obj


def read_json(path):
    with open(path, 'r') as stream:
        obj = json.load(stream, object_hook=_decode)
    return obj


def write_json(obj, path):
    with open(path, 'w') as stream:
        json.dump(obj, stream, default=_encode)

