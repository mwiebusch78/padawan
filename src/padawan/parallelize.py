import multiprocessing
import cloudpickle
import datetime


def _init_worker(func_and_args):
    global _func
    global _shared_args
    _func, _shared_args = cloudpickle.loads(func_and_args)


def _worker(x):
    return _func(x, **_shared_args)


def is_parallel_config(workers):
    if isinstance(workers, bool):
        return workers
    return workers < 0 or workers > 1


def _progress_message(msg, n):
    def progress_callback(completed, total, now, start):
        if completed % n != 0:
            return
        remaining = total - completed
        telapsed = now - start
        tremaining = telapsed/completed*remaining
        finish = now + tremaining
        print(
            msg.format(
                completed=completed,
                total=total,
                remaining=remaining,
                percent=int(completed/total*100),
                now=now.strftime('%Y-%m-%d %H:%M:%S'),
                start=start.strftime('%Y-%m-%d %H:%M:%S'),
                finish=finish.strftime('%Y-%m-%d %H:%M:%S'),
                telapsed=str(telapsed).split('.')[0],
                tremaining=str(tremaining).split('.')[0],
            )
        )

    return progress_callback


_DEFAULT_MSG = (
    'Completed {percent}% ({completed} of {total}). '
    'Time remaining: {tremaining}.'
)


def parallel_map(f, args, workers=False, shared_args=None, progress=False):
    if not workers:
        workers = False

    if isinstance(progress, str):
        progress = _progress_message(progress, 1)
    elif progress is True:
        progress = _progress_message(_DEFAULT_MSG, 1)
    elif progress is False:
        progress = None
    elif isinstance(progress, int):
        progress = _progress_message(_DEFAULT_MSG, progress)
    elif isinstance(progress, tuple):
        if len(progress) != 2 \
                or not isinstance(progress[0], str) \
                or not isinstance(progress[1], int):
            raise ValueError('Invalid value for progress argument.')
        progress = _progress_message(*progress)
    else:
        raise ValueError('Invalid value for progress argument.')

    if isinstance(workers, bool):
        workers = multiprocessing.cpu_count() if workers else 1
    workers = int(workers)
    if workers < 1:
        workers = multiprocessing.cpu_count() + workers

    if shared_args is None:
        shared_args = {}

    results = []
    num_items = len(args)
    t0 = datetime.datetime.now()
    if workers <= 1:
        for i, arg in enumerate(args, 1):
            results.append(f(arg, **shared_args))
            if progress is not None:
                progress(i, num_items, datetime.datetime.now(), t0)
    else:
        mp = multiprocessing.get_context(method='spawn')
        func_and_args = cloudpickle.dumps((f, shared_args))
        with mp.Pool(
                workers,
                initializer=_init_worker,
                initargs=(func_and_args,),
        ) as pool:
            results = []
            for i, result in enumerate(pool.imap(_worker, args), 1):
                results.append(result)
                if progress is not None:
                    progress(i, num_items, datetime.datetime.now(), t0)
            pool.close()
            pool.join()

    return results

