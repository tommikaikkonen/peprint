from .doc import (
    AlwaysBreak,
    Concat,
    Contextual,
    Doc,
    FlatChoice,
    Fill,
    Group,
    Nest,
    Annotated,
    NIL,
    LINE,
    SOFTLINE,
    HARDLINE,
)
from .utils import intersperse


def text(x):
    if not isinstance(x, str):
        raise TypeError("Argument to text function must be a str")
    return x


def cast_doc(doc):
    """Casts value to doc, if possible."""
    if isinstance(doc, Doc):
        return doc
    elif isinstance(doc, str):
        if doc == "":
            return NIL
        return doc

    raise ValueError(doc)


def group(doc):
    """Annotates doc with special meaning to the layout algorithm, so that the
    document is attempted to output on a single line if it is possible within
    the layout constraints. To lay out the doc on a single line, the `when_flat`
    branch of ``FlatChoice`` is used."""
    return Group(doc)


def concat(docs):
    """Returns a concatenation of the documents in the iterable argument"""
    docs = list(docs)
    if not docs:
        return NIL
    elif len(docs) == 1:
        return docs[0]
    return Concat([cast_doc(doc) for doc in docs])


def annotate(annotation, doc):
    """Annotates ``doc`` with the arbitrary value``annotation``"""
    return Annotated(doc, annotation)


def contextual(fn):
    """Returns a Doc that is lazily evaluated when deciding layout.

    ``fn`` must be a function that accepts four arguments:

    - ``indent`` (``int``): the current indentation level, 0 or more
    - ``column`` (``int``) the current output column in the output line
    - ``page_width`` (``int``) the requested page width (character count)
    - ``ribbon_width`` (``int``) the requested ribbon width (character count)
    """
    return Contextual(fn)


def align(doc):
    """Aligns each new line in ``doc`` with the first new line.
    """
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
    """Instructs the layout algorithm that ``doc`` must be
    broken to multiple lines. This instruction propagates
    to all higher levels in the layout, but nested Docs
    may still be laid out flat."""
    return AlwaysBreak(doc)


def flat_choice(when_broken, when_flat):
    """Gives the layout algorithm two options. ``when_flat`` Doc will be
    used when the document fit onto a single line, and ``when_broken`` is used
    when the Doc had to be broken into multiple lines."""
    return FlatChoice(when_broken, when_flat)
