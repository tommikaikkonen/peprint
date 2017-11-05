#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `peprint` package."""

import pytest
import datetime
import pytz
from itertools import cycle, islice
import json
import timeit

from hypothesis import given, settings
from hypothesis import strategies as st
from peprint import (
    pprint,
    pformat,
    cpprint,
)
from pprint import (
    pprint as nativepprint,
    pformat as nativepformat,
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


def test_align():
    doc = concat([
        text('lorem '),
        align(
            concat([
                'ipsum',
                HARDLINE,
                'aligned!'
            ])
        )
    ])

    expected = """\
lorem ipsum
      aligned!"""

    res = default_render_to_str(layout_smart(doc))
    assert res == expected


def test_fillsep():
    doc = fillsep(
        islice(
            cycle(["lorem", "ipsum", "dolor", "sit", "amet"]),
            20
        )
    )

    expected = """\
lorem ipsum dolor sit
amet lorem ipsum dolor
sit amet lorem ipsum
dolor sit amet lorem
ipsum dolor sit amet"""
    res = default_render_to_str(layout_smart(doc, width=20))
    assert res == expected


def test_always_breaking():
    """A dictionary value that is broken into multiple lines must
    also break the whole dictionary to multiple lines."""
    data = {
        'okay': ''.join(islice(cycle(['ab' * 20, ' ' * 3]), 3)),
    }
    expected = """\
{
    'okay':
        'abababababababababababababababababababab   '
        'abababababababababababababababababababab'
}"""
    res = pformat(data)
    assert res == expected


def test_pretty_json():
    with open('tests/sample_json.json') as f:
        data = json.load(f)

    print('native pprint')
    nativepprint(data)
    print('peprint')
    cpprint(data)


@pytest.mark.skip(reason="unskip to run performance test")
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
        st.binary() |
        st.datetimes() |
        st.timedeltas() |
        st.booleans() |
        st.just(None)
    )


hashable_primitives = (
    st.booleans() |
    st.integers() |
    st.floats(allow_nan=False) |
    st.text() |
    st.binary() |
    st.datetimes() |
    st.timedeltas()
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
            st.lists(base).map(tuple),
        )
    return st.recursive(primitives, extend)


def containers(primitives):
    def extend(base):
        return st.one_of(
            st.lists(base),
            st.lists(base).map(tuple),
            st.dictionaries(keys=hashable_containers(primitives), values=base),
        )

    return st.recursive(primitives, extend)


@given(containers(primitives()))
def test_all_python_values(value):
    cpprint(value)


@settings(max_examples=500, max_iterations=500)
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


@settings(max_examples=500, max_iterations=500)
@given(containers(primitives()))
def test_readable(value):
    formatted = pformat(value)

    assert eval(formatted, None, {'datetime': datetime}) == value


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
    expected = """\
[
    'ababababababababababababababababababababababababababababababababababa'
        'bababababababababababababababab',
    'ababababababababababababababababababababababababababababababababababa'
        'bababababababababababababababab'
]"""
    res = pformat(['ab' * 50] * 2)
    assert res == expected


def test_single_element_sequence_multiline_strategy():
    """Test that sequences with a single str-like element are not hang-indented
    in multiline mode."""
    expected = """\
[
    'ababababababababababababababababababababababababababababababababababa'
    'bababababababababababababababab'
]"""
    res = pformat(['ab' * 50])
    assert res == expected


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


def test_datetime():
    pprint(datetime.datetime.utcnow().replace(tzinfo=pytz.utc), width=40)
    pprint(datetime.timedelta(weeks=2, days=1, hours=3, milliseconds=5))
    neg_td = -datetime.timedelta(weeks=2, days=1, hours=3, milliseconds=5)
    pprint(neg_td)


@given(nested_dictionaries())
def test_nested_structures(value):
    pprint(value)
