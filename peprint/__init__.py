# -*- coding: utf-8 -*-

"""Top-level package for peprint."""

__author__ = """Tommi Kaikkonen"""
__email__ = 'kaikkonentommi@gmail.com'
__version__ = '0.1.0'

from pprint import isrecursive, isreadable, saferepr
from .peprint import pprint, pformat


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
