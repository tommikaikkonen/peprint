from enum import IntEnum, auto

class SyntaxIdentifier(IntEnum):
    ARGUMENTS = auto()
    KWARG_ASSIGN = auto()
    PREFIX = auto()
    IDENTIFIER = auto()
    IDENTIFIER_BUILTIN = auto()
    OPERATOR = auto()

    COMMA = auto()
    ELLIPSIS = auto()

    PAREN = auto()

    BRACKET = auto()

    BRACE = auto()

    SINGLETONS = auto()
    NUMBER_LITERAL = auto()
    STRING_LITERAL = auto()