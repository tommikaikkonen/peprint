from io import StringIO

from .sdoc import (
    SText,
    SLine,
    SMetaPush,
    SMetaPop,
)


def default_render_to_stream(stream, sdocs, newline='\n', separator=' '):
    evald = list(sdocs)
    for sdoc in evald:
        if isinstance(sdoc, str):
            stream.write(sdoc)
        elif isinstance(sdoc, SText):
            stream.write(sdoc.value)
        elif isinstance(sdoc, SLine):
            stream.write(newline + separator * sdoc.indent)


def default_render_to_str(sdocs, newline='\n', separator=' '):
    stream = StringIO()
    default_render_to_stream(stream, sdocs, newline, separator)
    return stream.getvalue()
