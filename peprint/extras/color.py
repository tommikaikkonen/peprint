import colorful
from ..sdoc import (
    SText,
    SLine,
    SMetaPush,
    SMetaPop,
)
from ..syntax import SyntaxIdentifier

_SYNTAX_IDENTIFIER_TO_COLOR = {
    SyntaxIdentifier.KWARG_ASSIGN: colorful.red,
    SyntaxIdentifier.ARGUMENTS: colorful.orange,
    SyntaxIdentifier.PREFIX: colorful.lightSkyBlue,
    SyntaxIdentifier.IDENTIFIER: colorful.reset,
    SyntaxIdentifier.IDENTIFIER_BUILTIN: colorful.cyan,
    SyntaxIdentifier.COMMA: colorful.reset,
    SyntaxIdentifier.ELLIPSIS: colorful.reset,
    SyntaxIdentifier.PAREN: colorful.reset,
    SyntaxIdentifier.BRACKET: colorful.reset,
    SyntaxIdentifier.BRACE: colorful.reset,
    SyntaxIdentifier.OPERATOR: colorful.red,
    SyntaxIdentifier.NUMBER_LITERAL: colorful.lightSkyBlue,
    SyntaxIdentifier.STRING_LITERAL: colorful.yellow,
    # None, True, False
    SyntaxIdentifier.SINGLETONS: colorful.hotPink,
}


def colored_render_to_stream(stream, sdocs, newline='\n', separator=' '):
    evald = list(sdocs)

    colorstack = []

    for sdoc in evald:
        if isinstance(sdoc, str):
            stream.write(sdoc)
        elif isinstance(sdoc, SText):
            stream.write(sdoc.value)
        elif isinstance(sdoc, SLine):
            stream.write(newline + separator * sdoc.indent)
        elif isinstance(sdoc, SMetaPush):
            if isinstance(sdoc.value, SyntaxIdentifier):
                color = _SYNTAX_IDENTIFIER_TO_COLOR[sdoc.value]
                colorstack.append(color)
                stream.write(str(color))

        elif isinstance(sdoc, SMetaPop):
            try:
                colorstack.pop()
            except IndexError:
                continue

            if colorstack:
                stream.write(str(colorstack[-1]))
            else:
                stream.write(str(colorful.reset))
