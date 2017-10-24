
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
