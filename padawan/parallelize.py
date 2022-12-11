import multiprocessing


def _init_worker(func, shared_args):
    global _func
    global _shared_args
    _func = func
    _shared_args = shared_args


def _worker(x):
    return _func(x, **_shared_args)


def parallel_map(f, args, workers=False, shared_args=None):
    if not workers:
        workers = False

    if isinstance(workers, bool):
        workers = multiprocessing.cpu_count() if workers else 1
    workers = int(workers)
    if workers < 1:
        workers = multiprocessing.cpu_count() + workers

    if shared_args is None:
        shared_args = {}

    if workers <= 1:
        results = []
        for arg in args:
            results.append(f(arg, **shared_args))
    else:
        mp = multiprocessing.get_context(method='spawn')
        with mp.Pool(
                workers,
                initializer=_init_worker,
                initargs=(f, shared_args),
        ) as pool:
            results = pool.map(_worker, args)
            pool.close()
            pool.join()

    return results

