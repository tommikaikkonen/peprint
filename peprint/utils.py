
def intersperse(x, ys):
    it = iter(ys)

    try:
        y = next(it)
    except StopIteration:
        return

    yield y

    for y in it:
        yield x
        yield y


def find(predicate, iterable, default=None):
    filtered = iter((x for x in iterable if predicate(x)))
    return next(filtered, default)


def rfind_idx(predicate, seq):
    length = len(seq)
    for i, el in enumerate(reversed(seq)):
        if predicate(el):
            return length - i - 1
    return -1
