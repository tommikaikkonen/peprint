def normalize_doc(doc):
    if isinstance(doc, str):
        return doc
    return doc.normalize()


def is_doc(doc):
    if isinstance(doc, str):
        return True
    return isinstance(doc, Doc)


class Doc:
    __slots__ = ()

    def normalize(self):
        return self


class WithMeta(Doc):
    __slots__ = ('doc', 'meta')

    def __init__(self, doc, meta):
        self.doc = doc
        self.meta = meta

    def normalize(self):
        if self.meta is None:
            return self.doc
        return self

    def __repr__(self):
        return f'Identity({repr(self.doc)})'


class Text(Doc):
    __slots__ = ('value', )

    def __init__(self, value):
        if not isinstance(value, str):
            raise TypeError(
                f"Got {repr(value)} of type {type(value).__name__}, "
                "expected 'str'"
            )
        self.value = value

    def __repr__(self):
        return f'Text({repr(self.value)})'


class Nil(Doc):
    def __repr__(self):
        return 'NIL'


NIL = Nil()


class Concat(Doc):
    __slots__ = ('docs', )

    def __init__(self, docs):
        self.docs = docs

    def normalize(self):
        normalized_docs = []
        propagate_broken = False
        for doc in self.docs:
            doc = normalize_doc(doc)
            if isinstance(doc, Concat):
                normalized_docs.extend(doc.docs)
            elif isinstance(doc, AlwaysBreak):
                propagate_broken = True
                normalized_docs.append(doc.doc)
            else:
                normalized_docs.append(doc)

        res = Concat(normalized_docs)
        if propagate_broken:
            res = AlwaysBreak(res)
        return res

    def __repr__(self):
        return f"Concat({', '.join(repr(doc) for doc in self.docs)})"


class Nest(Doc):
    __slots__ = ('indent', 'doc')

    def __init__(self, indent, doc):
        assert isinstance(indent, int)
        assert isinstance(doc, Doc)

        self.indent = indent
        self.doc = doc

    def normalize(self):
        inner_normalized = normalize_doc(self.doc)
        if isinstance(inner_normalized, AlwaysBreak):
            return AlwaysBreak(
                Nest(self.indent, inner_normalized.doc)
            )
        return Nest(self.indent, normalize_doc(self.doc))

    def __repr__(self):
        return f'Nest({repr(self.indent)}, {repr(self.doc)})'


class FlatChoice(Doc):
    __slots__ = ('when_broken', 'when_flat')

    def __init__(self, when_broken, when_flat):
        self.when_broken = when_broken
        self.when_flat = when_flat

    def normalize(self):
        broken_normalized = normalize_doc(self.when_broken)
        if isinstance(broken_normalized, AlwaysBreak):
            return broken_normalized

        flat_normalized = normalize_doc(self.when_flat)
        if isinstance(broken_normalized, AlwaysBreak):
            return flat_normalized

        return FlatChoice(
            broken_normalized,
            flat_normalized
        )

    def __repr__(self):
        return (
            f'FlatChoice(when_broken={repr(self.when_broken)}, '
            f'when_flat={repr(self.when_flat)})'
        )


class Contextual(Doc):
    __slots__ = ('fn', )

    def __init__(self, fn):
        self.fn = fn

    def __repr__(self):
        return f'Contextual({repr(self.fn)})'


class HardLine(Doc):
    def __repr__(self):
        return 'HardLine()'


HARDLINE = HardLine()
LINE = FlatChoice(HARDLINE, Text(' '))
SOFTLINE = FlatChoice(HARDLINE, NIL)


class Group(Doc):
    __slots__ = ('doc', )

    def __init__(self, doc):
        assert isinstance(doc, Doc)
        self.doc = doc

    def normalize(self):
        doc_normalized = normalize_doc(self.doc)
        if isinstance(doc_normalized, AlwaysBreak):
            # Group is the possibility of either flat
            # or break; since we're always breaking,
            # we don't need Group.
            return doc_normalized
        return Group(doc_normalized)

    def __repr__(self):
        return f'Group({repr(self.doc)})'


class AlwaysBreak(Doc):
    __slots__ = ('doc', )

    def __init__(self, doc):
        assert isinstance(doc, Doc)
        self.doc = doc

    def normalize(self):
        doc_normalized = normalize_doc(self.doc)
        if isinstance(doc_normalized, AlwaysBreak):
            return doc_normalized
        return AlwaysBreak(doc_normalized)

    def __repr__(self):
        return f'AlwaysBreak({repr(self.doc)})'


class Fill(Doc):
    __slots__ = ('docs', )

    def __init__(self, docs):
        self.docs = list(docs)

    def normalize(self):
        return Fill([normalize_doc(doc) for doc in self.docs])

    def __repr__(self):
        return f"Fill([{', '.join(repr(doc) for doc in self.docs)}])"
