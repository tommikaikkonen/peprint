#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `peprint` package."""

import pytest
from itertools import cycle, islice
import math
import json
import timeit

from hypothesis import given, settings, seed
from hypothesis import strategies as st
from peprint import pprint, pformat
from pprint import (
    pprint as nativepprint,
    pformat as nativepformat
)
from peprint.api import (
    align,
    concat,
    fillsep,
    text,
    HARDLINE
)
from peprint.render import default_render_to_str
from peprint.layout import layout_smart


def test_content():
    pprint([i for i in range(10)])
    val = {
        'yas': True,
        'okay': {
            'que': 1,
            'wut': "yeasdfweafsfwefmkvamkwmfaldkfmalkemlfaksmdlkamlskmalwkemlakmdflkamwlekfm asdf kmwe amwkdfm awlefk masdfkl awe ",
            'wyd': True,
        },
    }
    pprint(val, width=79)
    nativepprint(val, width=79)

    val = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        1,
        2,
        3,
        {'a': 2, 'b': 3}
    ]
    pprint(val)
    nativepprint(val)

    pprint(('kewlio', ))
    nativepprint(('kewlio', ))

    pprint("asdfom ad fwekrj asdf jwerakjsdfna wr nasfj akwer akjsdf jlkawjer asf mnvaker nakjdfn kawe rkajdfn kajwenr awer asdf alwekr asdf aknwmen kawjern aisudfn kawek an")
    assert False


def test_align():
    doc = concat([
        text('example '),
        align(
            concat([
                'okay',
                HARDLINE,
                'aligned!'
            ])
        )
    ])
    print(default_render_to_str(layout_smart(doc)))
    assert False


def test_fillsep():
    doc = fillsep(islice(cycle(["lorem", "ipsum", "dolor", "sit", "amet"]), 50))
    print(default_render_to_str(layout_smart(doc)))
    assert False


def test_always_breaking():
    data = {
        'okay': ''.join(islice(cycle(['ab' * 20, ' ' * 3]), 10)),
    }
    pprint(data)
    assert False


def test_pretty_json():
    with open('tests/sample_json.json') as f:
        data = json.load(f)

    print('native pprint')
    nativepprint(data)
    print('peprint')
    pprint(data)
    assert False


def test_perf():
    with open('tests/sample_json.json') as f:
        data = json.load(f)

    print('native pprint')
    native_dur = timeit.timeit(
        'nativepformat(data)',
        globals={
            'nativepformat': nativepformat,
            'data': data,
        },
        number=500,
    )
    # nativepprint(data, depth=2)
    print('peprint')
    peprint_dur = timeit.timeit(
        'pformat(data)',
        globals={
            'pformat': pformat,
            'data': data,
        },
        number=500,
    )

    print(f'Native pprint took {native_dur}, peprint took {peprint_dur}')
    assert False


def test_recursive():
    d = {}
    d['self_recursion'] = d

    rendered = pformat(d)
    assert rendered == f"{{'self_recursion': <Recursion on dict with id={id(d)}>}}"


def primitives():
    return (
        st.integers() |
        st.floats(allow_nan=False) |
        st.text() |
        st.binary()
    )


hashable_primitives = (
    st.booleans() |
    st.integers() |
    st.floats(allow_nan=False) |
    st.text() |
    st.binary()
)


def identity(x):
    return x


def hashables():
    def extend(base):
        return base.flatmap(
            lambda strat: st.tuples(
                strat,
                st.sampled_from([
                    st.tuples,
                    st.frozensets,
                ])
            )
        ).map(lambda strat__extend: strat__extend[1](strat__extend[0]))

    return st.recursive(hashable_primitives, extend)


def hashable_containers(primitives):
    def extend(base):
        return st.one_of(
            st.frozensets(base),
            st.tuples(base)
        )
    return st.recursive(primitives, extend)


def containers(primitives):
    def extend(base):
        return st.one_of(
            st.lists(base),
            st.tuples(base),
            st.dictionaries(keys=hashable_containers(primitives), values=base),
        )

    return st.recursive(primitives, extend)


@given(containers(primitives()))
def test_all_python_values(value):
    pprint(value)


@settings(max_examples=5000, max_iterations=5000)
@given(st.binary())
def test_bytes_pprint_equals_repr(bytestr):
    reprd = repr(bytestr)
    pformatted = pformat(bytestr)

    # This is not always the case. E.g.:
    # >>> print(repr(b"\"''"))
    # >>> b'"\'\''
    #
    # Where as peprint chooses
    # >>> print(pformat(b"\"''""))
    # >>> b"\"''"
    # For fewer escapes.
    used_same_quote_type = reprd[-1] == pformatted[-1]

    if used_same_quote_type:
        assert pformat(bytestr) == repr(bytestr)


@settings(max_examples=5000, max_iterations=5000)
@given(containers(primitives()))
def test_readable(value):
    formatted = pformat(value)
    assert eval(formatted) == value


def nested_dictionaries():
    simple_strings_alphabet = 'abcdefghijklmnopqrstuvwxyz\'"\r\n '
    simple_text = st.text(alphabet=simple_strings_alphabet, min_size=5, average_size=20)

    def extend(base):
        return st.one_of(
            st.lists(base, min_size=5),
            st.dictionaries(keys=simple_text, values=base, min_size=1)
        )

    return st.recursive(simple_text, extend, max_leaves=50)


def test_top_level_str():
    """Tests that top level strings are not indented or surrounded with parentheses"""

    pprint('ab' * 50)
    expected = (
        "'ababababababababababababababababababababababababababababababababababa'"
        "\n'bababababababababababababababab'"
    )
    assert pformat('ab' * 50) == expected


def test_second_level_str():
    """Test that second level strs are indented"""
    pprint({'ab' * 50: 'cd' * 100})
    expected = """\
[
    'ababababababababababababababababababababababababababababababababababa'
        'bababababababababababababababab'
]"""
    assert pformat(['ab' * 50]) == expected


def test_many_cases():
    # top-level multiline str.
    pprint('abcd' * 40)

    # sequence with multiline strs.
    pprint(['abcd' * 40] * 5)

    # nested dict
    pprint({
        'ab' * 40: 'cd' * 50
    })

    # long urls.
    pprint([
        'https://www.example.com/User/john/files/Projects/peprint/images/original/image0001.jpg'
            '?q=verylongquerystring&maxsize=1500&signature=af429fkven2aA'
            '#content1-header-something-something'
    ] * 5)
    nativepprint([
        'https://www.example.com/User/john/files/Projects/peprint/images/original/image0001.jpg'
            '?q=verylongquerystring&maxsize=1500&signature=af429fkven2aA'
            '#content1-header-something-something'
    ] * 5)

@given(nested_dictionaries())
def test_nested_structures(value):
    pprint(value)
