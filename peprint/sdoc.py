
class SDoc(object):
    pass


class SNil(SDoc):
    pass


SNIL = SNil()


class SLine(SDoc):
    __slots__ = ('indent', )

    def __init__(self, indent):
        assert isinstance(indent, int)
        self.indent = indent

    def __repr__(self):
        return f'SLine({repr(self.indent)})'


class SAnnotationPush(SDoc):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'SAnnotationPush({repr(self.value)})'


class SAnnotationPop(SDoc):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'SAnnotationPush({repr(self.value)})'
