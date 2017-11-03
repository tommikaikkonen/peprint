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
    WithMeta,
    NIL,
    LINE,
    SOFTLINE,
    HARDLINE,
)
from .utils import intersperse


def text(x):
    return x


def cast_doc(doc):
    if isinstance(doc, Doc):
        return doc
    elif isinstance(doc, str):
        if doc == "":
            return NIL
        return doc

    raise ValueError(doc)


def group(x):
    return Group(x)


def concat(xs):
    xs = list(xs)
    if not xs:
        return NIL
    elif len(xs) == 1:
        return xs[0]
    return Concat([cast_doc(x) for x in xs])


def with_meta(meta, doc):
    return WithMeta(doc, meta)


def contextual(fn):
    return Contextual(fn)


def align(doc):
    def evaluator(indent, column, page_width, ribbon_width):
        return Nest(column - indent, doc)
    return contextual(evaluator)


def hang(i, doc):
    return align(Nest(i, doc))


def nest(i, doc):
    return Nest(i, doc)


def hsep(docs):
    return concat(intersperse(' ', docs))


def vsep(docs):
    return concat(intersperse(LINE, docs))


def fillsep(docs):
    return Fill(intersperse(LINE, map(cast_doc, docs)))


def fill(docs):
    return Fill(docs)


def always_break(doc):
    return AlwaysBreak(doc)


def flat_choice(when_broken, when_flat):
    return FlatChoice(when_broken, when_flat)
