
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


class SText(SDoc):
    __slots__ = ('value', )

    def __init__(self, value):
        assert isinstance(value, str)
        self.value = value

    def __repr__(self):
        return f'SText({repr(self.value)})'


class SMetaPush(SDoc):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'SMetaPush({repr(self.value)})'


class SMetaPop(SDoc):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'SMetaPush({repr(self.value)})'
