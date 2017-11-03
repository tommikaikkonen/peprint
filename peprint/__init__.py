# -*- coding: utf-8 -*-

"""Top-level package for peprint."""

__author__ = """Tommi Kaikkonen"""
__email__ = 'kaikkonentommi@gmail.com'
__version__ = '0.1.0'

from io import StringIO
import sys

from pprint import isrecursive, isreadable, saferepr
from .peprint import python_to_sdocs
from .render import default_render_to_stream


class PrettyPrinter:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def pprint(self, object):
        pprint(*self._args, **self._kwargs)

    def pformat(self, object):
        return pformat(*self._args, **self._kwargs)

    def isrecursive(self, object):
        return isrecursive(object)

    def isreadable(self, object):
        return isreadable(object)

    def format(self, object):
        raise NotImplementedError


def pformat(object, indent=4, width=79, depth=None, *, compact=False):
    # TODO: compact
    sdocs = python_to_sdocs(object, indent=indent, width=width, depth=depth)
    stream = StringIO()
    default_render_to_stream(stream, sdocs)
    return stream.getvalue()


def pprint(object, stream=None, indent=4, width=79, depth=None, *, compact=False, end='\n'):
    # TODO: compact
    sdocs = python_to_sdocs(object, indent=indent, width=width, depth=depth)
    if stream is None:
        stream = sys.stdout
    default_render_to_stream(stream, sdocs)
    if end:
        stream.write(end)


try:
    from .extras.color import colored_render_to_stream
except ImportError:
    def cpprint(*args, **kwargs):
        raise ImportError("You need to install the 'colorful' package for colored output.")
else:
    def cpprint(
        object,
        stream=None,
        indent=4,
        width=79,
        depth=None,
        *,
        compact=False,
        style=None,
        end='\n'
    ):
        sdocs = python_to_sdocs(object, indent=indent, width=width, depth=depth)
        if stream is None:
            stream = sys.stdout
        colored_render_to_stream(stream, sdocs, style=style)
        if end:
            stream.write(end)


def install_to_ipython():
    try:
        import IPython.lib.pretty
    except ImportError:
        return

    try:
        ipy = get_ipython()
    except NameError:
        return

    class IPythonCompatPrinter:
        def __init__(self, stream, *args, **kwargs):
            self.stream = stream

        def pretty(self, obj):
            style = ipy.highlighting_style
            if style == 'legacy':
                # Fall back to default
                style = None

            cpprint(obj, stream=self.stream, style=style, end=None)

        def flush(self):
            pass

    IPython.lib.pretty.RepresentationPrinter = IPythonCompatPrinter
