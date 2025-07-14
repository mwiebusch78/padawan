_DEFAULT_MSG = (
    'Completed {percent}% ({completed} of {total}). '
    'Time remaining: {tremaining}.'
)


def _progress_message(msg, n):
    def progress_callback(completed, total, now, start):
        if not msg:
            return
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


def make_progress_callback(progress):
    if isinstance(progress, str):
        progress = _progress_message(progress, 1)
    elif progress is True:
        progress = _progress_message(_DEFAULT_MSG, 1)
    elif progress is False:
        progress = _progress_message(None, 1)
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
    return progress
