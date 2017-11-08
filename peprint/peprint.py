import math
import re
from collections import Counter, OrderedDict
from datetime import (
    datetime,
    timedelta,
    tzinfo,
    timezone,
    date,
    time,
)
from functools import singledispatch, partial
from itertools import chain, cycle, dropwhile
from uuid import UUID

from .api import (
    concat,
    contextual,
    nest,
    always_break,
    group,
    annotate,
    NIL,
    LINE,
    SOFTLINE,
    HARDLINE
)

from .layout import layout_smart
from .syntax import Token
from .utils import intersperse

try:
    import pytz
except ImportError:
    _PYTZ_INSTALLED = False
else:
    _PYTZ_INSTALLED = True

UNSET_SENTINEL = object()

COMMA = annotate(Token.PUNCTUATION, ',')
COLON = annotate(Token.PUNCTUATION, ':')
ELLIPSIS = annotate(Token.PUNCTUATION, '...')

LPAREN = annotate(Token.PUNCTUATION, '(')
RPAREN = annotate(Token.PUNCTUATION, ')')

LBRACKET = annotate(Token.PUNCTUATION, '[')
RBRACKET = annotate(Token.PUNCTUATION, ']')

LBRACE = annotate(Token.PUNCTUATION, '{')
RBRACE = annotate(Token.PUNCTUATION, '}')

NEG_OP = annotate(Token.OPERATOR, '-')
MUL_OP = annotate(Token.OPERATOR, '*')
ADD_OP = annotate(Token.OPERATOR, '+')


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


def builtin_identifier(s):
    return annotate(Token.NAME_BUILTIN, s)


def identifier(s):
    return annotate(Token.NAME_FUNCTION, s)


def general_identifier(s):
    if callable(s):
        module, qualname = s.__module__, s.__qualname__

        if module == 'builtins':
            return builtin_identifier(qualname)
        return identifier(f'{module}.{qualname}')
    return identifier(s)


class PrettyContext:
    __slots__ = (
        'indent',
        'depth_left',
        'visited',
        'multiline_strategy',
        'user_ctx'
    )

    def __init__(
        self,
        indent,
        depth_left,
        visited=None,
        multiline_strategy=MULTILINE_STATEGY_PLAIN,
        user_ctx=None,
    ):
        self.indent = indent
        self.depth_left = depth_left
        self.multiline_strategy = multiline_strategy

        if visited is None:
            visited = set()
        self.visited = visited

        if user_ctx is None:
            user_ctx = {}

        self.user_ctx = user_ctx

    def _replace(self, **kwargs):
        passed_keys = set(kwargs.keys())
        fieldnames = type(self).__slots__
        assert passed_keys.issubset(set(fieldnames))
        return PrettyContext(
            **{
                k: (
                    kwargs[k]
                    if k in passed_keys
                    else getattr(self, k)
                )
                for k in fieldnames
            }
        )

    def use_multiline_strategy(self, strategy):
        return self._replace(multiline_strategy=strategy)

    def set(self, key, value):
        return self._replace(user_ctx={
            **self.user_ctx,
            key: value,
        })

    def get(self, key, default=None):
        return self.user_ctx.get(key, default)

    def nested_call(self):
        return self._replace(depth_left=self.depth_left - 1)

    def start_visit(self, value):
        self.visited.add(id(value))

    def end_visit(self, value):
        self.visited.remove(id(value))

    def is_visited(self, value):
        return id(value) in self.visited


def _pretty_dispatch(produce_doc, value, ctx):
    if ctx.is_visited(value):
        return _pretty_recursion(value)

    ctx.start_visit(value)

    doc = produce_doc(value, ctx)

    ctx.end_visit(value)

    return doc


def _repr_pretty(value, ctx):
    return repr(value)


pretty_python_value = singledispatch(partial(_pretty_dispatch, _repr_pretty))


def register_pretty(_type):
    def decorator(fn):
        pretty_python_value.register(_type, partial(_pretty_dispatch, fn))
        return fn
    return decorator


def bracket(ctx, left, child, right):
    return concat([
        left,
        nest(ctx.indent, concat([SOFTLINE, child])),
        SOFTLINE,
        right
    ])


def comment(text):
    return annotate(Token.COMMENT_SINGLE, concat(['# ', text]))


def sequence_of_docs(ctx, left, docs, right, dangle=False):
    sep = concat([COMMA, LINE])
    separated_els = intersperse(sep, docs)

    if dangle:
        separated_els = chain(separated_els, [COMMA])

    return group(bracket(ctx, left, concat(separated_els), right))


def prettycall(ctx, fn, *args, **kwargs):
    fndoc = general_identifier(fn)

    if ctx.depth_left <= 0:
        return concat([fndoc, LPAREN, ELLIPSIS, RPAREN])

    if not kwargs and len(args) == 1:
        sole_arg = args[0]
        if isinstance(sole_arg, (list, dict, tuple)):
            return build_fncall(
                ctx,
                fndoc,
                argdocs=[pretty_python_value(sole_arg, ctx)],
                hug_sole_arg=True,
            )

    nested_ctx = (
        ctx
        .nested_call()
        .use_multiline_strategy(MULTILINE_STATEGY_HANG)
    )

    return build_fncall(
        ctx,
        fndoc,
        argdocs=(
            pretty_python_value(arg, nested_ctx)
            for arg in args
        ),
        kwargdocs=(
            (kwarg, pretty_python_value(v, nested_ctx))
            for kwarg, v in kwargs.items()
        ),
    )


def build_fncall(
    ctx,
    fndoc,
    argdocs=(),
    kwargdocs=(),
    hug_sole_arg=False
):
    """Builds a doc that looks like a function call,
    from docs that represent the function, arguments
    and keyword arguments.

    If ``hug_sole_arg`` is True, and the represented
    functional call is done with a single non-keyword
    argument, the function call parentheses will hug
    the sole argument doc without newlines and indentation
    in break mode. This makes a difference in calls
    like this:

    > hug_sole_arg = False
    frozenset(
        [
            1,
            2,
            3,
            4,
            5
        ]
    )
    > hug_sole_arg = True
    frozenset([
        1,
        2,
        3,
        4,
        5,
    ])
    """
    if callable(fndoc):
        fndoc = general_identifier(fndoc)

    argsep = concat([COMMA, LINE])

    kwargdocs = [
        concat([
            binding,
            annotate(
                Token.OPERATOR,
                '='
            ),
            doc
        ])
        for binding, doc in kwargdocs
    ]

    argdocs = list(argdocs)
    if not (argdocs or kwargdocs):
        return concat([
            fndoc,
            LPAREN,
            RPAREN,
        ])

    if hug_sole_arg and not kwargdocs and len(argdocs) == 1:
        return group(
            concat([
                fndoc,
                LPAREN,
                argdocs[0],
                RPAREN
            ])
        )

    allarg_docs = [*argdocs, *kwargdocs]

    return group(
        concat([
            fndoc,
            LPAREN,
            nest(
                ctx.indent,
                concat([
                    SOFTLINE,
                    concat(
                        intersperse(
                            argsep,
                            allarg_docs
                        )
                    ),
                ])
            ),
            SOFTLINE,
            RPAREN
        ])
    )


@register_pretty(tuple)
@register_pretty(list)
@register_pretty(set)
def pretty_bracketable_iterable(value, ctx):
    dangle = False

    if isinstance(value, list):
        left, right = LBRACKET, RBRACKET
    elif isinstance(value, tuple):
        left, right = LPAREN, RPAREN
        if len(value) == 1:
            dangle = True
    elif isinstance(value, set):
        left, right = LBRACE, RBRACE

    if not value:
        if isinstance(value, (list, tuple)):
            return concat([left, right])
        else:
            assert isinstance(value, set)
            return prettycall(ctx, set)

    if ctx.depth_left == 0:
        return concat([left, ELLIPSIS, right])

    if len(value) == 1:
        sole_value = list(value)[0]
        els = [
            pretty_python_value(
                sole_value,
                ctx=(
                    ctx
                    .nested_call()
                    .use_multiline_strategy(MULTILINE_STATEGY_PLAIN)
                )
            )
        ]
    else:
        els = (
            pretty_python_value(
                el,
                ctx=(
                    ctx
                    .nested_call()
                    .use_multiline_strategy(MULTILINE_STATEGY_HANG)
                )
            )
            for el in value
        )

    return sequence_of_docs(ctx, left, els, right, dangle=dangle)


@register_pretty(frozenset)
def pretty_frozenset(value, ctx):
    if value:
        return prettycall(ctx, frozenset, list(value))
    return prettycall(ctx, frozenset)


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
def pretty_dict(d, ctx):
    if ctx.depth_left == 0:
        return concat([LBRACE, ELLIPSIS, RBRACE])
    pairs = []
    for k in sorted(d.keys(), key=_AlwaysSortable):
        v = d[k]

        if isinstance(k, (str, bytes)):
            kdoc = pretty_str(
                k,
                # not a nested call on purpose
                ctx=ctx.use_multiline_strategy(MULTILINE_STATEGY_PARENS),
            )
        else:
            kdoc = pretty_python_value(
                k,
                ctx=ctx.nested_call()
            )

        vdoc = pretty_python_value(
            v,
            ctx=(
                ctx
                .nested_call()
                .use_multiline_strategy(MULTILINE_STATEGY_INDENTED)
            ),
        )
        pairs.append((kdoc, vdoc))

    pairsep = concat([COMMA, LINE])

    pairdocs = (
        group(
            concat([
                kdoc,
                concat([COLON, ' ']),
                vdoc
            ]),
        )
        for kdoc, vdoc in pairs
    )

    doc = bracket(
        ctx,
        LBRACE,
        concat(intersperse(pairsep, pairdocs)),
        RBRACE,
    )

    if len(pairs) > 2:
        doc = always_break(doc)
    else:
        doc = group(doc)

    return doc


@register_pretty(Counter)
def pretty_counter(counter, ctx):
    return prettycall(ctx, Counter, dict(counter))


INF_FLOAT = float('inf')
NEG_INF_FLOAT = float('-inf')


@register_pretty(float)
def pretty_float(value, ctx):
    if ctx.depth_left == 0:
        return prettycall(ctx, float, ...)

    if value == INF_FLOAT:
        return prettycall(ctx, float, 'inf')
    elif value == NEG_INF_FLOAT:
        return prettycall(ctx, float, '-inf')
    elif math.isnan(value):
        return prettycall(ctx, float, 'nan')

    return annotate(Token.NUMBER_FLOAT, repr(value))


@register_pretty(int)
def pretty_int(value, ctx):
    if ctx.depth_left == 0:
        return prettycall(ctx, int, ...)
    return annotate(Token.NUMBER_INT, repr(value))


@register_pretty(type(...))
def pretty_ellipsis(value, ctx):
    return ELLIPSIS


@register_pretty(bool)
@register_pretty(type(None))
def pretty_singletons(value, ctx):
    return annotate(Token.KEYWORD_CONSTANT, repr(value))


SINGLE_QUOTE_TEXT = "'"
SINGLE_QUOTE_BYTES = b"'"

DOUBLE_QUOTE_TEXT = '"'
DOUBLE_QUOTE_BYTES = b'"'


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
        return SINGLE_QUOTE_TEXT

    if not contains_double:
        return DOUBLE_QUOTE_TEXT

    assert contains_single and contains_double

    single_count = s.count(single_quote)
    double_count = s.count(double_quote)

    if single_count <= double_count:
        return SINGLE_QUOTE_TEXT

    return DOUBLE_QUOTE_TEXT


def escape_str_for_quote(use_quote, s):
    escaped_with_quotes = repr(s)
    repr_used_quote = escaped_with_quotes[-1]

    # string may have a prefix
    first_quote_at_index = escaped_with_quotes.find(repr_used_quote)
    repr_escaped = escaped_with_quotes[first_quote_at_index + 1:-1]

    if repr_used_quote == use_quote:
        # repr produced the quotes we wanted -
        # escaping is correct.
        return repr_escaped

    # repr produced different quotes, which escapes
    # alternate quotes.
    if use_quote == SINGLE_QUOTE_TEXT:
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


def pretty_single_line_str(s, indent, use_quote=None):
    prefix = (
        annotate(Token.STRING_AFFIX, 'b')
        if isinstance(s, bytes)
        else ''
    )

    if use_quote is None:
        use_quote = determine_quote_strategy(s)

    return concat([
        prefix,
        annotate(
            Token.LITERAL_STRING,
            concat([
                use_quote,
                escape_str_for_quote(use_quote, s),
                use_quote
            ])
        )
    ])


def split_at(idx, sequence):
    return (sequence[:idx], sequence[idx:])


def escaped_len(s, use_quote):
    return len(escape_str_for_quote(use_quote, s))


WHITESPACE_PATTERN_TEXT = re.compile(r'(\s+)')
WHITESPACE_PATTERN_BYTES = re.compile(rb'(\s+)')

NONWORD_PATTERN_TEXT = re.compile(r'(\W+)')
NONWORD_PATTERN_BYTES = re.compile(rb'(\W+)')


def str_to_lines(max_len, use_quote, s):
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
        starts_with_whitespace = bool(nonword_pattern.match(alternating_words_ws[0]))
    else:
        starts_with_whitespace = bool(whitespace_pattern.match(alternating_words_ws[0]))

    # List[Tuple[str, bool]]
    # The boolean associated with each part indicates if it is a
    # whitespce/non-word part or not.
    tagged_alternating = list(
        zip(
            alternating_words_ws,
            cycle([starts_with_whitespace, not starts_with_whitespace])
        )
    )

    remaining_stack = list(reversed(tagged_alternating))
    curr_line_parts = []
    while remaining_stack:
        curr, is_whitespace = remaining_stack.pop()
        curr_line_parts.append(curr)
        curr_line_len = sum(
            escaped_len(part, use_quote)
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

            remaining_len = max_len - (curr_line_len - escaped_len(curr, use_quote))
            this_line_part, next_line_part = split_at(max(remaining_len, 0), curr)

            curr_line_parts.append(this_line_part)

            yield empty.join(curr_line_parts)
            curr_line_parts = []

            if next_line_part:
                remaining_stack.append((next_line_part, is_whitespace))

    if curr_line_parts:
        yield empty.join(curr_line_parts)


@register_pretty(str)
@register_pretty(bytes)
def pretty_str(s, ctx):
    if ctx.depth_left == 0:
        if isinstance(s, str):
            return prettycall(ctx, str, ...)
        else:
            assert isinstance(s, bytes)
            return prettycall(ctx, bytes, ...)

    multiline_strategy = ctx.multiline_strategy
    peprint_indent = ctx.indent

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

        use_quote = determine_quote_strategy(s)

        lines = str_to_lines(
            max_len=each_line_max_str_len,
            use_quote=use_quote,
            s=s,
        )

        parts = intersperse(
            HARDLINE,
            (
                pretty_single_line_str(
                    line,
                    indent=peprint_indent,
                    use_quote=use_quote,
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
                left_paren, right_paren = LPAREN, RPAREN
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
                    (
                        HARDLINE
                        if multiline_strategy == MULTILINE_STATEGY_PARENS
                        else NIL
                    ),
                    right_paren
                ])
            )

    return contextual(evaluator)


@register_pretty(UUID)
def pretty_uuid(value, ctx):
    return prettycall(ctx, UUID, str(value))


@register_pretty(datetime)
def pretty_datetime(dt, ctx):
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
        (
            k,
            pretty_python_value(v, ctx.nested_call())
        )
        for k, v in reversed(kwargs_to_show)
    ]

    if dt.tzinfo is not None:
        tzinfodoc = pretty_python_value(
            dt.tzinfo,
            ctx=ctx.nested_call()
        )
        kwargdocs.append(
            ('tzinfo', tzinfodoc)
        )

    if dt.fold:
        kwargdocs.append(('fold', pretty_python_value(1, ctx=ctx)))

    if len(kwargdocs) == 3:  # year, month, day
        return prettycall(
            ctx,
            datetime,
            dt.year,
            dt.month,
            dt.day,
        )

    return group(
        build_fncall(
            ctx,
            datetime,
            kwargdocs=kwargdocs,
        )
    )


@register_pretty(tzinfo)
def pretty_tzinfo(value, ctx):
    if value == timezone.utc:
        return identifier('datetime.timezone.utc')
    elif _PYTZ_INSTALLED and value == pytz.utc:
        return identifier('pytz.utc')


@register_pretty(timezone)
def pretty_timezone(tz, ctx):
    if tz == timezone.utc:
        return identifier('datetime.timezone.utc')

    if tz._name is None:
        return prettycall(ctx, timezone, tz._offset)
    return prettycall(ctx, timezone, tz._offset, tz._name)


def pretty_pytz_timezone(tz, ctx):
    if tz == pytz.utc:
        return identifier('pytz.utc')
    return prettycall(ctx, pytz.timezone, tz.zone)


def pretty_pytz_dst_timezone(tz, ctx):
    if pytz.timezone(tz.zone) == tz:
        return pretty_pytz_timezone(tz, ctx)

    # timezone can't be represented with a
    # pytz.timezone(zonename) call.
    # TODO: output comments that have
    # the zone name.
    return prettycall(
        ctx,
        pytz.tzinfo.DstTzInfo,
        (tz._utcoffset, tz._dst, tz._tzname)
    )


if _PYTZ_INSTALLED:
    register_pretty(pytz.tzinfo.BaseTzInfo)(pretty_pytz_timezone)
    register_pretty(pytz.tzinfo.DstTzInfo)(pretty_pytz_dst_timezone)


@register_pretty(time)
def pretty_time(value, ctx):
    timekws_to_display = reversed(
        list(
            dropwhile(
                lambda kw: getattr(value, kw) == 0,
                ('microsecond', 'second', 'minute', 'hour')
            )
        )
    )

    additional_kws = []
    if value.tzinfo is not None:
        additional_kws.append(('tzinfo', value.tzinfo))

    if value.fold != 0:
        additional_kws.append(('fold', value.fold))

    kwargs = chain(
        (
            (kw, getattr(value, kw))
            for kw in timekws_to_display
        ),
        additional_kws
    )

    return prettycall(
        ctx,
        time,
        **OrderedDict(kwargs)
    )


@register_pretty(date)
def pretty_date(value, ctx):
    return prettycall(ctx, date, value.year, value.month, value.day)


@register_pretty(timedelta)
def pretty_timedelta(delta, ctx):
    if ctx.depth_left == 0:
        return prettycall(ctx, timedelta, ...)

    pos_delta = abs(delta)
    negative = delta != pos_delta

    days = pos_delta.days
    seconds = pos_delta.seconds
    microseconds = pos_delta.microseconds

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds, microseconds = divmod(microseconds, 1000)

    attrs = [
        ('days', days),
        ('hours', hours),
        ('minutes', minutes),
        ('seconds', seconds),
        ('milliseconds', milliseconds),
        ('microseconds', microseconds),
    ]

    kwargdocs = [
        (k, pretty_python_value(v, ctx=ctx.nested_call()))
        for k, v in attrs
        if v != 0
    ]

    if kwargdocs and kwargdocs[0][0] == 'days':
        years, days = divmod(days, 365)
        if years:
            _docs = []

            if years > 1:
                _docs.extend([
                    pretty_python_value(years, ctx),
                    ' ',
                    MUL_OP,
                    ' '
                ])

            _docs.append(pretty_python_value(365, ctx))

            if days:
                _docs.extend([
                    ' ',
                    ADD_OP,
                    ' ',
                    pretty_python_value(days, ctx)
                ])

            kwargdocs[0] = ('days', concat(_docs))

    doc = group(
        build_fncall(
            ctx,
            timedelta,
            kwargdocs=kwargdocs,
        )
    )

    if negative:
        doc = concat([NEG_OP, doc])

    return doc


@register_pretty(OrderedDict)
def pretty_ordereddict(d, ctx):
    return prettycall(ctx, OrderedDict, [*d.items()])


def _pretty_recursion(value):
    return f'<Recursion on {type(value).__name__} with id={id(value)}>'


def python_to_sdocs(value, indent, width, depth, ribbon_width=71):
    if depth is None:
        depth = float('inf')

    doc = pretty_python_value(
        value,
        ctx=PrettyContext(indent=indent, depth_left=depth, visited=set())
    )

    ribbon_frac = min(1.0, ribbon_width / width)

    return layout_smart(doc, width=width, ribbon_frac=ribbon_frac)
