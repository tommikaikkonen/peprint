from .doc import (
    AlwaysBreak,
    Concat,
    Contextual,
    Doc,
    FlatChoice,
    Fill,
    Group,
    Nest,
    Text,
    NIL,
    LINE,
    SOFTLINE,
    HARDLINE,
)
from .utils import intersperse


def text(x, *, meta=None):
    if meta is None:
        return x
    return Text(x, meta=meta)


def cast_doc(doc):
    if isinstance(doc, Doc):
        return doc
    elif isinstance(doc, str):
        if doc == "":
            return NIL
        return doc

    raise ValueError(doc)


def group(x, *, meta=None):
    return Group(x, meta=meta)


def concat(xs, *, meta=None):
    xs = list(xs)
    if not xs:
        return NIL
    elif len(xs) == 1:
        return xs[0]
    return Concat([cast_doc(x) for x in xs], meta=meta)


def contextual(fn, *, meta=None):
    return Contextual(fn, meta=meta)


def align(doc, *, meta=None):
    def evaluator(indent, column, page_width, ribbon_width):
        return Nest(column - indent, doc)
    return contextual(evaluator, meta=meta)


def hang(i, doc, *, meta=None):
    return align(Nest(i, doc), meta=meta)


def nest(i, doc, *, meta=None):
    return Nest(i, doc, meta=meta)


def hsep(docs, *, meta=None):
    return concat(intersperse(' ', docs), meta=meta)


def vsep(docs, *, meta=None):
    return concat(intersperse(LINE, docs), meta=meta)


def fillsep(docs, *, meta=None):
    return Fill(intersperse(LINE, map(cast_doc, docs)), meta=meta)


def fill(docs, *, meta=None):
    return Fill(docs, meta=meta)


def always_break(doc, *, meta=None):
    return AlwaysBreak(doc, meta=meta)


def flat_choice(when_broken, when_flat, *, meta=None):
    return FlatChoice(when_broken, when_flat, meta=meta)
