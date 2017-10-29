import enum
import math
import re
import sys
from datetime import datetime, timedelta
from functools import singledispatch, partial
from io import StringIO
from itertools import chain, cycle, dropwhile

from .api import (
    concat,
    contextual,
    nest,
    always_break,
    flat_choice,
    group,
    LINE,
    SOFTLINE,
    HARDLINE
)
from .doc import is_doc

from .layout import layout_smart
from .render import default_render_to_stream

from .utils import intersperse


def _pretty_dispatch(produce_doc, value, indent, depth_left, visited=None):
    if visited is None:
        visited = set()

    assert isinstance(visited, set)

    value_id = id(value)
    if value_id in visited:
        return _pretty_recursion(value)
    else:
        visited.add(value_id)

    doc = produce_doc(value, indent, depth_left, visited=visited)

    visited.remove(value_id)

    return doc


def _repr_pretty(value, indent, depth_left, visited=None):
    return repr(value)


pretty_python_value = singledispatch(partial(_pretty_dispatch, _repr_pretty))


def register_pretty(_type):
    def decorator(fn):
        pretty_python_value.register(_type, partial(_pretty_dispatch, fn))
        return fn
    return decorator


def write_doc_to_stream(doc, stream, width, newline='\n', separator=' '):
    sdocs = layout_smart(doc, width=width)
    default_render_to_stream(
        stream,
        sdocs,
        newline=newline,
        separator=separator
    )


def bracket(left, child, right, indent, force_break=False,):
    outer = always_break if force_break else group
    return outer(
        concat([
            left,
            nest(indent, concat([SOFTLINE, child])),
            SOFTLINE,
            right
        ])
    )


def fncall(fndoc, argdocs=(), kwargdocs=(), indent=None):
    if indent is None:
        raise ValueError

    if not is_doc(fndoc):
        fndoc = f'{fndoc.__module__}.{fndoc.__qualname__}'

    argsep = concat([',', LINE])

    kwargdocs = (
        concat([f'{binding}=', doc])
        for binding, doc in kwargdocs
    )

    return concat([
        fndoc,
        nest(
            indent,
            concat([
                '(',
                SOFTLINE,
                concat(
                    intersperse(
                        argsep,
                        chain(argdocs, kwargdocs)
                    )
                ),
            ])
        ),
        SOFTLINE,
        ')'
    ])


@register_pretty(set)
@register_pretty(frozenset)
def pretty_bracketable_iterable(value, indent, depth_left, visited):
    if isinstance(value, list):
        left, right = '[', ']'
    elif isinstance(value, tuple):
        left, right = '(', ')'
    elif isinstance(value, set):
        left, right = '{', '}'
    elif isinstance(value, frozenset):
        left, right = 'frozenset({', '})'

    if depth_left == 0:
        return f'{left}...{right}'

    sep = concat([',', LINE])
    els = (
        (
            pretty_str(
                el,
                indent=indent,
                depth_left=depth_left - 1,
                multiline_strategy=MULTILINE_STATEGY_HANG,
                visited=visited,
            )
            if isinstance(el, (str, bytes))
            else pretty_python_value(
                el,
                indent=indent,
                depth_left=depth_left - 1,
                visited=visited,
            )
        )
        for el in value
    )

    return bracket(
        left,
        concat(intersperse(sep, els)),
        right,
        indent=indent
    )


@register_pretty(list)
def pretty_list(arr, indent, depth_left, visited):
    return pretty_bracketable_iterable(
        arr,
        indent=indent,
        depth_left=depth_left,
        visited=visited,
    )


@register_pretty(tuple)
def pretty_tuple(tup, indent, depth_left, visited):
    if depth_left == 0:
        return 'tuple(...)'

    if not tup:
        return '()'
    elif len(tup) == 1:
        return concat([
            '(',
            pretty_python_value(
                tup[0],
                indent=indent,
                depth_left=depth_left - 1,
                visited=visited,
            ),
            ',)'
        ])
    else:
        return pretty_bracketable_iterable(tup)


class _AlwaysSortable(object):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def sortable_value(self):
        return (str(type(self)), id(self))

    def __lt__(self, other):
        try:
            return self.value < other.value
        except TypeError:
            return self.sortable_value() < other.sortable_value()


@register_pretty(dict)
def pretty_dict(d, indent, depth_left, visited):
    if depth_left == 0:
        return '{...}'
    pairs = []
    for k in sorted(d.keys(), key=_AlwaysSortable):
        v = d[k]

        if isinstance(k, (str, bytes)):
            kdoc = pretty_str(
                k,
                indent=indent,
                depth_left=depth_left - 1,
                multiline_strategy=MULTILINE_STATEGY_PARENS,
                visited=visited,
            )
        else:
            kdoc = pretty_python_value(
                k,
                indent=indent,
                depth_left=depth_left - 1
            )

        if isinstance(v, (str, bytes)):
            vdoc = pretty_str(
                v,
                indent=indent,
                depth_left=depth_left - 1,
                multiline_strategy=MULTILINE_STATEGY_INDENTED,
                visited=visited,
            )
        else:
            vdoc = pretty_python_value(
                v,
                indent=indent,
                depth_left=depth_left - 1,
                visited=visited,
            )
        pairs.append((kdoc, vdoc))

    pairsep = concat([',', LINE])

    pairdocs = (
        group(
            concat([
                kdoc,
                ': ',
                vdoc
            ]),
        )
        for kdoc, vdoc in pairs
    )

    res = bracket(
        '{',
        concat(intersperse(pairsep, pairdocs)),
        '}',
        indent=4,
        force_break=len(pairs) > 2,
    )

    return res


INF_FLOAT = float('inf')
NEG_INF_FLOAT = float('-inf')


@register_pretty(float)
def pretty_float(value, indent, depth_left, visited):
    if depth_left == 0:
        return 'float(...)'

    if value == INF_FLOAT:
        return "float('inf')"
    elif value == NEG_INF_FLOAT:
        return "float('-inf')"
    elif math.isnan(value):
        return "float('nan')"

    return repr(value)


SINGLE_QUOTE_TEXT = "'"
SINGLE_QUOTE_BYTES = b"'"

DOUBLE_QUOTE_TEXT = '"'
DOUBLE_QUOTE_BYTES = b'"'


class QuoteStrategy(enum.Enum):
    SINGLE = SINGLE_QUOTE_TEXT
    DOUBLE = DOUBLE_QUOTE_TEXT

    @property
    def bytes_quote(self):
        if self.value == SINGLE_QUOTE_TEXT:
            return SINGLE_QUOTE_BYTES
        return DOUBLE_QUOTE_BYTES

    @property
    def text_quote(self):
        return self.value


def determine_quote_strategy(s):
    if isinstance(s, str):
        single_quote = SINGLE_QUOTE_TEXT
        double_quote = DOUBLE_QUOTE_TEXT
    else:
        single_quote = SINGLE_QUOTE_BYTES
        double_quote = DOUBLE_QUOTE_BYTES

    contains_single = single_quote in s
    contains_double = double_quote in s

    if not contains_single:
        return QuoteStrategy.SINGLE

    if not contains_double:
        return QuoteStrategy.DOUBLE

    assert contains_single and contains_double

    single_count = s.count(single_quote)
    double_count = s.count(double_quote)

    if single_count <= double_count:
        return QuoteStrategy.SINGLE

    return QuoteStrategy.DOUBLE


def escape_str_for_quote(quote_strategy, s):
    if not isinstance(quote_strategy, QuoteStrategy):
        raise ValueError(f'Expected QuoteStrategy, got {quote_strategy}')

    escaped_with_quotes = repr(s)
    repr_used_quote = escaped_with_quotes[-1]

    # string may have a prefix
    first_quote_at_index = escaped_with_quotes.find(repr_used_quote)
    repr_escaped = escaped_with_quotes[first_quote_at_index + 1:-1]

    if repr_used_quote == quote_strategy.text_quote:
        # repr produced the quotes we wanted -
        # escaping is correct.
        return repr_escaped

    # repr produced different quotes, which escapes
    # alternate quotes.
    if quote_strategy == QuoteStrategy.SINGLE:
        # repr used double quotes
        return (
            repr_escaped
            .replace('\\"', DOUBLE_QUOTE_TEXT)
            .replace(SINGLE_QUOTE_TEXT, "\\'")
        )
    else:
        # repr used single quotes
        return (
            repr_escaped
            .replace("\\'", SINGLE_QUOTE_TEXT)
            .replace(DOUBLE_QUOTE_TEXT, '\\"')
        )


def pretty_single_line_str(s, indent, quote_strategy=None):
    prefix = (
        'b'
        if isinstance(s, bytes)
        else ''
    )

    if quote_strategy is None:
        quote_strategy = determine_quote_strategy(s)
    else:
        if not isinstance(quote_strategy, QuoteStrategy):
            raise ValueError

    quote = quote_strategy.text_quote

    return concat([
        prefix,
        quote,
        escape_str_for_quote(quote_strategy, s),
        quote
    ])


def split_at(idx, sequence):
    return (sequence[:idx], sequence[idx:])


def escaped_len(s, quote_strategy):
    return len(escape_str_for_quote(quote_strategy, s))


WHITESPACE_PATTERN_TEXT = re.compile(r'(\s+)')
WHITESPACE_PATTERN_BYTES = re.compile(rb'(\s+)')

NONWORD_PATTERN_TEXT = re.compile(r'(\W+)')
NONWORD_PATTERN_BYTES = re.compile(rb'(\W+)')


def str_to_lines(max_len, quote_strategy, s):
    if isinstance(s, str):
        whitespace_pattern = WHITESPACE_PATTERN_TEXT
        nonword_pattern = NONWORD_PATTERN_TEXT
        empty = ''
    else:
        assert isinstance(s, bytes)
        whitespace_pattern = WHITESPACE_PATTERN_BYTES
        nonword_pattern = NONWORD_PATTERN_BYTES
        empty = b''

    alternating_words_ws = whitespace_pattern.split(s)

    if len(alternating_words_ws) <= 1:
        # no whitespace: try splitting with nonword pattern.
        alternating_words_ws = nonword_pattern.split(s)
        starts_with_whitespace = nonword_pattern.match(alternating_words_ws[0])
    else:
        starts_with_whitespace = whitespace_pattern.match(alternating_words_ws[0])

    tagged_alternating = (
        list(zip(alternating_words_ws, cycle([True, False])))
        if starts_with_whitespace
        else list(zip(alternating_words_ws, cycle([False, True])))
    )

    remaining_stack = list(reversed(tagged_alternating))
    curr_line_parts = []
    while remaining_stack:
        curr, is_whitespace = remaining_stack.pop()
        curr_line_parts.append(curr)
        curr_line_len = sum(
            escaped_len(part, quote_strategy)
            for part in curr_line_parts
        )

        if curr_line_len == max_len:
            if not is_whitespace and len(curr_line_parts) > 2:
                curr_line_parts.pop()
                yield empty.join(curr_line_parts)
                curr_line_parts = []
                remaining_stack.append((curr, is_whitespace))
            else:
                yield empty.join(curr_line_parts)
                curr_line_parts = []
                continue
        elif curr_line_len > max_len:
            if not is_whitespace and len(curr_line_parts) > 1:
                curr_line_parts.pop()
                yield empty.join(curr_line_parts)
                remaining_stack.append((curr, is_whitespace))
                curr_line_parts = []
                continue

            curr_line_parts.pop()

            remaining_len = max_len - (curr_line_len - escaped_len(curr, quote_strategy))
            this_line_part, next_line_part = split_at(max(remaining_len, 0), curr)

            curr_line_parts.append(this_line_part)

            yield empty.join(curr_line_parts)
            curr_line_parts = []

            if next_line_part:
                remaining_stack.append((next_line_part, is_whitespace))

    if curr_line_parts:
        yield empty.join(curr_line_parts)



# For dict keys
"""
(
    'aaaaaaaaaa'
    'aaaaaa'
)
"""
MULTILINE_STATEGY_PARENS = 'MULTILINE_STATEGY_PARENS'

# For dict values
"""
    'aaaaaaaaaa'
    'aaaaa'
"""
MULTILINE_STATEGY_INDENTED = 'MULTILINE_STATEGY_INDENTED'

# For sequence elements
"""
'aaaaaaaaa'
    'aaaaaa'
"""
MULTILINE_STATEGY_HANG = 'MULTILINE_STATEGY_HANG'

# For top level strs
"""
'aaaaaaaaa'
'aaaaaa'
"""
MULTILINE_STATEGY_PLAIN = 'MULTILINE_STATEGY_PLAIN'


@register_pretty(str)
@register_pretty(bytes)
def pretty_str(
    s,
    indent,
    depth_left,
    visited,
    multiline_strategy=MULTILINE_STATEGY_PLAIN,
):
    if depth_left == 0:
        if isinstance(s, str):
            return 'str(...)'
        else:
            assert isinstance(s, bytes)
            return 'bytes(...)'

    peprint_indent = indent

    def evaluator(indent, column, page_width, ribbon_width):
        columns_left_in_line = page_width - column
        columns_left_in_ribbon = indent + ribbon_width - column
        available_width = min(columns_left_in_line, columns_left_in_ribbon)

        singleline_str_chars = len(s) + len('""')
        flat_version = pretty_single_line_str(s, peprint_indent)

        if singleline_str_chars <= available_width:
            return flat_version

        # multiline string
        each_line_starts_on_col = indent + peprint_indent
        each_line_ends_on_col = min(page_width, each_line_starts_on_col + ribbon_width)

        each_line_max_str_len = each_line_ends_on_col - each_line_starts_on_col - 2

        quote_strategy = determine_quote_strategy(s)

        lines = str_to_lines(
            max_len=each_line_max_str_len,
            quote_strategy=quote_strategy,
            s=s,
        )

        parts = intersperse(
            HARDLINE,
            (
                pretty_single_line_str(
                    line,
                    indent=peprint_indent,
                    quote_strategy=quote_strategy,
                )
                for line in lines
            )
        )

        if multiline_strategy == MULTILINE_STATEGY_PLAIN:
            return always_break(concat(parts))
        elif multiline_strategy == MULTILINE_STATEGY_HANG:
            return always_break(
                nest(
                    peprint_indent,
                    concat(parts)
                )
            )
        else:
            if multiline_strategy == MULTILINE_STATEGY_PARENS:
                left_paren, right_paren = '(', ')'
            else:
                assert multiline_strategy == MULTILINE_STATEGY_INDENTED
                left_paren, right_paren = '', ''

            return always_break(
                concat([
                    left_paren,
                    nest(
                        peprint_indent,
                        concat([
                            HARDLINE,
                            *parts,
                        ])
                    ),
                    HARDLINE,
                    right_paren
                ])
            )

    return contextual(evaluator)


def _pretty_datetime_flat(dt, indent, depth_left, visited):
    return repr(dt)


def _pretty_datetime_broken(dt, indent, depth_left, visited):
    dt_kwargs = [
        (k, getattr(dt, k))
        for k in (
            'microsecond',
            'second',
            'minute',
            'hour',
            'day',
            'month',
            'year',
        )
    ]

    kwargs_to_show = list(
        dropwhile(
            lambda k__v: k__v[1] == 0,
            dt_kwargs
        )
    )

    kwargdocs = [
        # values are always ints, so we can shortcut
        # by returning a repr str instead of calling
        # pretty_python_value
        (k, repr(v))
        for k, v in reversed(kwargs_to_show)
    ]

    if dt.tzinfo is not None:
        tzinfodoc = pretty_python_value(
            dt.tzinfo,
            indent=indent,
            depth_left=depth_left - 1,
            visited=visited,
        )
        kwargdocs.append(
            ('tzinfo', tzinfodoc)
        )

    if dt.fold:
        kwargdocs.append(('fold', '1'))

    return fncall(
        datetime,
        kwargdocs=kwargdocs,
        indent=indent
    )


@register_pretty(datetime)
def pretty_datetime(dt, indent, depth_left, visited):
    return group(
        flat_choice(
            when_flat=_pretty_datetime_flat(
                dt,
                indent,
                depth_left=depth_left,
                visited=visited,
            ),
            when_broken=_pretty_datetime_broken(
                dt,
                indent,
                depth_left=depth_left,
                visited=visited,
            )
        )
    )


@register_pretty(timedelta)
def pretty_timedelta(delta, indent, depth_left, visited):
    pos_delta = abs(delta)
    negative = delta != pos_delta

    days = pos_delta.days
    seconds = pos_delta.seconds
    microseconds = pos_delta.microseconds

    weeks, days = divmod(days, 7)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds, microseconds = divmod(microseconds, 1000)

    attrs = [
        ('weeks', weeks),
        ('days', days),
        ('hours', hours),
        ('minutes', minutes),
        ('seconds', seconds),
        ('milliseconds', milliseconds),
        ('microseconds', microseconds),
    ]

    return group(
        fncall(
            f"{'-' if negative else ''}datetime.timedelta",
            kwargdocs=(
                (k, repr(v))
                for k, v in attrs
                if v != 0
            ),
            indent=indent
        )
    )


def _pretty_recursion(value):
    return f'<Recursion on {type(value).__name__} with id={id(value)}>'


def pformat(object, indent=4, width=79, depth=None, *, compact=False):
    stream = StringIO()
    if depth is None:
        depth = float('inf')
    doc = pretty_python_value(
        object,
        indent=indent,
        depth_left=depth,
        visited=set()
    )
    # TODO: depth
    # TODO: compact
    write_doc_to_stream(doc, stream, width=width, separator=' ')
    return stream.getvalue()


def pprint(object, stream=None, indent=4, width=79, depth=None, *, compact=False):
    if depth is None:
        depth = float('inf')

    doc = pretty_python_value(
        object,
        indent=indent,
        depth_left=depth,
        visited=set()
    )
    if stream is None:
        stream = sys.stdout

    # TODO: depth
    # TODO: compact
    write_doc_to_stream(doc, stream, width=width, separator=' ')
    stream.write('\n')
