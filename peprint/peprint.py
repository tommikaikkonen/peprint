from collections import namedtuple
from itertools import cycle
from io import StringIO
import math
import re
import sys

from .api import (
    concat,
    contextual,
    nest,
    always_break,
    group,
    LINE,
    SOFTLINE,
    HARDLINE
)
from .layout import layout_smart
from .render import default_render_to_stream

from .utils import intersperse


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


def pretty_bracketable_iterable(value, indent, depth_left=None):
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
            pretty_str(el, indent=indent, depth_left=depth_left - 1, parentheses=True)
            if isinstance(el, (str, bytes))
            else pretty_python_value(el, indent=indent, depth_left=depth_left - 1)
        )
        for el in value
    )

    return bracket(
        left,
        concat(intersperse(sep, els)),
        right,
        indent=indent
    )


def pretty_list(arr, indent, depth_left):
    return pretty_bracketable_iterable(
        arr,
        indent=indent,
        depth_left=depth_left
    )


def pretty_tuple(tup, indent, depth_left):
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
                depth_left=depth_left - 1
            ),
            ',)'
        ])
    else:
        return pretty_bracketable_iterable(tup)


def pretty_dict(d, indent, depth_left):
    if depth_left == 0:
        return '{...}'
    pairs = []
    for k, v in d.items():
        if isinstance(k, (str, bytes)):
            kdoc = pretty_str(
                k,
                indent=indent,
                depth_left=depth_left - 1,
                parentheses=True,
                requires_further_indent=True,
            )
        else:
            kdoc = pretty_python_value(
                k,
                indent=indent,
                depth_left=depth_left - 1
            )

        vdoc = pretty_python_value(
            v,
            indent=indent,
            depth_left=depth_left - 1,
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


def pretty_float(value, indent, depth_left):
    if value == INF_FLOAT:
        return "float('inf')"
    elif value == NEG_INF_FLOAT:
        return "float('-inf')"
    elif math.isnan(value):
        return "float('nan')"
    return repr(value)


SINGLE_QUOTE = "'"
DOUBLE_QUOTE = '"'


StringRenderingApproach = namedtuple(
    'StringRenderingApproach',
    'quote_type, needs_escaping'
)


def needs_escaping_for_quote(quote_type, s):
    return quote_type in s


def determine_string_rendering_approach(s):
    if isinstance(s, str):
        contains_single = SINGLE_QUOTE in s
        contains_double = DOUBLE_QUOTE in s
    else:
        contains_single = SINGLE_QUOTE.encode('ascii') in s
        contains_double = DOUBLE_QUOTE.encode('ascii') in s

    if not contains_single:
        return StringRenderingApproach(SINGLE_QUOTE, needs_escaping=False)

    if not contains_double:
        return StringRenderingApproach(DOUBLE_QUOTE, needs_escaping=False)

    assert contains_single and contains_double

    if isinstance(s, str):
        single_count = s.count(SINGLE_QUOTE)
        double_count = s.count(DOUBLE_QUOTE)
    else:
        single_count = s.count(SINGLE_QUOTE.encode('ascii'))
        double_count = s.count(DOUBLE_QUOTE.encode('ascii'))

    if single_count <= double_count:
        return StringRenderingApproach(SINGLE_QUOTE, needs_escaping=True)

    return StringRenderingApproach(DOUBLE_QUOTE, needs_escaping=True)


def escape_str_for_quote(quote, s):
    assert quote in (SINGLE_QUOTE, DOUBLE_QUOTE)

    escaped_with_quotes = repr(s)
    repr_used_quote = escaped_with_quotes[-1]
    # string may have a prefix
    first_quote_at_index = escaped_with_quotes.find(repr_used_quote)
    repr_escaped = escaped_with_quotes[first_quote_at_index + 1:-1]

    if repr_used_quote == quote:
        # repr produced the quotes we wanted -
        # escaping is correct.
        return repr_escaped

    # repr produced different quotes, which escapes
    # alternate quotes.
    if quote == SINGLE_QUOTE:
        # repr used double quotes
        return (
            repr_escaped
            .replace('\\"', DOUBLE_QUOTE)
            .replace(SINGLE_QUOTE, "\\'")
        )
    else:
        # repr used single quotes
        return (
            repr_escaped
            .replace("\\'", SINGLE_QUOTE)
            .replace(DOUBLE_QUOTE, '\\"')
        )


def pretty_single_line_str(s, indent, quote_type=None, strtype=str):
    prefix = (
        'b'
        if strtype is bytes
        else ''
    )

    if quote_type is None:
        rendering_approach = determine_string_rendering_approach(s)
        quote_type = rendering_approach.quote_type

    return concat([
        prefix,
        quote_type,
        escape_str_for_quote(quote_type, s),
        quote_type
    ])


def split_at(idx, sequence):
    return (sequence[:idx], sequence[idx:])


def escaped_len(s, quote_type):
    return len(escape_str_for_quote(quote_type, s))


MAX_ESCAPE_LEN = len('\\U123456789')


def str_to_lines(max_len, quote_type, s):
    if isinstance(s, str):
        empty = ''
        alternating_words_ws = re.split(r'(\s+)', s)
        starts_with_whitespace = re.match(r'\s+', alternating_words_ws[0])
    else:
        empty = b''
        alternating_words_ws = re.split(rb'(\s+)', s)
        starts_with_whitespace = re.match(rb'\s+', alternating_words_ws[0])

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
            escaped_len(part, quote_type)
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

            remaining_len = max_len - (curr_line_len - escaped_len(curr, quote_type))
            this_line_part, next_line_part = split_at(max(remaining_len, 0), curr)

            curr_line_parts.append(this_line_part)

            yield empty.join(curr_line_parts)
            curr_line_parts = []

            if next_line_part:
                # remaining_len is calculated from an escaped str, which
                # is longer than unescaped `curr`. E.g:
                # >>> len('\U000108390')
                # 1
                # >>> len('\\U000108390')
                # 11
                # therefore next_line_part may be empty.
                remaining_stack.append((next_line_part, is_whitespace))

    if curr_line_parts:
        yield empty.join(curr_line_parts)


def pretty_str(
    s,
    indent,
    depth_left,
    parentheses=False,
    requires_further_indent=True,
):
    strtype = (
        str
        if isinstance(s, str)
        else bytes
    )

    peprint_indent = indent

    if parentheses:
        left_paren, right_paren = '(', ')'
    else:
        left_paren, right_paren = '', ''

    def evaluator(indent, column, page_width, ribbon_width):
        columns_left_in_line = page_width - column
        columns_left_in_ribbon = indent + ribbon_width - column
        available_width = min(columns_left_in_line, columns_left_in_ribbon)

        singleline_str_chars = len(s) + len('""')
        flat_version = pretty_single_line_str(s, peprint_indent, strtype=strtype)

        if singleline_str_chars <= available_width:
            return flat_version

        # multiline string
        each_line_starts_on_col = indent + peprint_indent
        each_line_ends_on_col = min(page_width, each_line_starts_on_col + ribbon_width)

        each_line_max_str_len = each_line_ends_on_col - each_line_starts_on_col - 2

        if strtype is str:
            count_single_quotes = s.count(SINGLE_QUOTE)
            count_double_quotes = s.count(DOUBLE_QUOTE)
        else:
            count_single_quotes = s.count(SINGLE_QUOTE.encode('ascii'))
            count_double_quotes = s.count(DOUBLE_QUOTE.encode('ascii'))

        if not (count_single_quotes or count_double_quotes):
            quote_type = SINGLE_QUOTE
        elif count_single_quotes < count_double_quotes:
            quote_type = SINGLE_QUOTE
        else:
            quote_type = DOUBLE_QUOTE

        lines = str_to_lines(
            max_len=each_line_max_str_len,
            quote_type=quote_type,
            s=s,
        )

        parts = intersperse(
            HARDLINE,
            (
                pretty_single_line_str(
                    line,
                    indent=peprint_indent,
                    quote_type=quote_type,
                    strtype=strtype
                )
                for line in lines
            )
        )

        if parentheses:
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

        return always_break(
            nest(
                (
                    peprint_indent
                    if requires_further_indent
                    else 0
                ),
                concat([
                    HARDLINE,
                    *parts
                ])
            ),
        )

    return contextual(evaluator)


DELEGATE_TO_REPR_PRIMITIVES = {
    type(None),
    int,
    bool,
}


def _pretty_recursion(value):
    return f'<Recursion on {type(value).__name__} with id={id(value)}>'


def pretty_python_value(value, indent, depth_left=float('inf'), visited=set()):
    value_id = id(value)
    if value_id in visited:
        return _pretty_recursion(value)
    else:
        visited.add(value_id)

    if type(value) in DELEGATE_TO_REPR_PRIMITIVES:
        doc = repr(value)
    elif isinstance(value, str):
        doc = pretty_str(
            value,
            indent=indent,
            depth_left=depth_left
        )
    elif isinstance(value, float):
        doc = pretty_float(
            value,
            indent=indent,
            depth_left=depth_left
        )
    elif isinstance(value, tuple):
        doc = pretty_tuple(
            value,
            indent=indent,
            depth_left=depth_left
        )
    elif isinstance(value, (list, set, frozenset)):
        doc = pretty_bracketable_iterable(
            value,
            indent=indent,
            depth_left=depth_left
        )
    elif isinstance(value, dict):
        doc = pretty_dict(
            value,
            indent=indent,
            depth_left=depth_left
        )
    elif isinstance(value, bytes):
        doc = pretty_str(
            value,
            indent=indent,
            depth_left=depth_left,
        )
    elif isinstance(value, complex):
        doc = repr(value)
    else:
        # TODO: there's more types
        # to be handled here.
        doc = repr(value)

    visited.remove(value_id)

    return doc


def pformat(object, indent=4, width=79, depth=None, *, compact=False):
    stream = StringIO()
    if depth is None:
        depth = float('inf')
    doc = pretty_python_value(object, indent=indent, depth_left=depth)
    # TODO: depth
    # TODO: compact
    write_doc_to_stream(doc, stream, width=width, separator=' ')
    return stream.getvalue()


def pprint(object, stream=None, indent=4, width=79, depth=None, *, compact=False):

    if depth is None:
        depth = float('inf')

    doc = pretty_python_value(object, indent=indent, depth_left=depth)

    if stream is None:
        stream = sys.stdout

    # TODO: depth
    # TODO: compact
    write_doc_to_stream(doc, stream, width=width, separator=' ')
    stream.write('\n')
