import multiprocessing
import cloudpickle


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
        func_and_args = cloudpickle.dumps((f, shared_args))
        with mp.Pool(
                workers,
                initializer=_init_worker,
                initargs=(func_and_args,),
        ) as pool:
            results = pool.map(_worker, args)
            pool.close()
            pool.join()

    return results

