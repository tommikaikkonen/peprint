from ..sdoc import (
    SText,
    SLine,
    SMetaPush,
    SMetaPop,
)
from ..syntax import Token

_COLOR_DEPS_INSTALLED = True
try:
    from pygments import token
    from pygments import styles
except ImportError:
    _COLOR_DEPS_INSTALLED = False
else:
    _SYNTAX_TOKEN_TO_PYGMENTS_TOKEN = {
        Token.KEYWORD_CONSTANT: token.Keyword.Constant,
        Token.NAME_BUILTIN: token.Name.Builtin,
        Token.NAME_ENTITY: token.Name.Entity,
        Token.NAME_FUNCTION: token.Name.Function,
        Token.LITERAL_STRING: token.String,
        Token.STRING_AFFIX: token.String.Affix,
        Token.STRING_ESCAPE: token.String.Escape,
        Token.NUMBER_INT: token.Number,
        Token.NUMBER_BINARY: token.Number.Bin,
        Token.NUMBER_INT: token.Number.Integer,
        Token.NUMBER_FLOAT: token.Number.Float,
        Token.OPERATOR: token.Operator,
        Token.PUNCTUATION: token.Punctuation,
    }

    default_style = styles.get_style_by_name('monokai')

try:
    import colorful
except ImportError:
    _COLOR_DEPS_INSTALLED = False


def styleattrs_to_colorful(attrs):
    c = colorful.reset
    if attrs['color'] or attrs['bgcolor']:
        # Colorful doesn't have a way to directly set Hex/RGB
        # colors- until I find a better way, we do it like this :)
        accessor = ''
        if attrs['color']:
            colorful.update_palette({'peprintCurrFg': attrs['color']})
            accessor = 'peprintCurrFg'
        if attrs['bgcolor']:
            colorful.update_palette({'peprintCurrBg': attrs['bgcolor']})
            accessor += '_on_peprintCurrBg'
        c &= getattr(colorful, accessor)
    if attrs['bold']:
        c &= colorful.bold
    if attrs['italic']:
        c &= colorful.italic
    if attrs['underline']:
        c &= colorful.underline
    return c


def colored_render_to_stream(stream, sdocs, style, newline='\n', separator=' '):
    if not _COLOR_DEPS_INSTALLED:
        raise Exception(
            "'pygments' and 'colorful' packages must be "
            "installed to use colored output."
        )

    if style is None:
        style = default_style

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
            if isinstance(sdoc.value, Token):
                pygments_token = _SYNTAX_TOKEN_TO_PYGMENTS_TOKEN[sdoc.value]
                tokenattrs = style.style_for_token(pygments_token)
                color = styleattrs_to_colorful(tokenattrs)
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

    if colorstack:
        stream.write(str(colorful.reset))
