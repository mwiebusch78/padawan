import multiprocessing
import cloudpickle
import datetime

from .progress import make_progress_callback


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


def parallel_map(f, args, workers=False, shared_args=None, progress=False):
    if not workers:
        workers = False

    progress = make_progress_callback(progress)

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
                progress(i, num_items, datetime.datetime.now(), t0)
            pool.close()
            pool.join()

    return results

