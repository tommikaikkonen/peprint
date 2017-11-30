import inspect
import math
import re
from functools import singledispatch, partial
from itertools import chain, cycle

from .api import (
    always_break,
    annotate,
    concat,
    contextual,
    flat_choice,
    fill,
    group,
    nest,
    NIL,
    LINE,
    SOFTLINE,
    HARDLINE
)
from .doc import (
    Annotated,
    Doc
)

from .layout import layout_smart
from .syntax import Token
from .utils import identity, intersperse

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
ASSIGN_OP = annotate(Token.OPERATOR, '=')

WHITESPACE_PATTERN_TEXT = re.compile(r'(\s+)')
WHITESPACE_PATTERN_BYTES = re.compile(rb'(\s+)')

NONWORD_PATTERN_TEXT = re.compile(r'(\W+)')
NONWORD_PATTERN_BYTES = re.compile(rb'(\W+)')


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


IMPLICIT_MODULES = {
    '__main__',
    'builtins',
}


class CommentAnnotation:
    def __init__(self, value):
        assert isinstance(value, str)
        self.value = value

    def __repr__(self):
        return f'ValueComment({repr(self.value)})'


class _CommentedValue:
    def __init__(self, value, comment):
        self.value = value
        self.comment = comment


class _TrailingCommentedValue:
    def __init__(self, value, comment):
        self.value = value
        self.comment = comment


def annotate_comment(comment, doc):
    """Annotate ``doc`` with ``comment`` text.

    Peprint will make sure the parent (or top-level) handler
    will render the comment in a proper way. E.g. if ``doc``
    represents an element in a list, then the ``list`` pretty
    printer will handle where to place the comment.

    Differs from ``comment`` and ``trailing_comment`` by
    operating only on Docs, not normal values.
    """
    return annotate(CommentAnnotation(comment), doc)


def comment(comment_str, value):
    """Annotates a value with a comment str.

    Allows you to insert comments into Peprint output
    by annotating them on the values directly, instead
    of first having to render them into a Doc and then
    annotating the Doc with ``annotate_comment``.

    Generally, you want to use this to annotate arguments
    to ``prettycall``.
    """
    return _CommentedValue(value, comment_str)


def trailing_comment(comment_str, value):
    """Annotates a value with a comment str, so that
    the comment will be rendered "trailing", e.g. in place
    of the last element in a list, set or tuple, or after
    the last argument in a function.

    This will force the rendering of `value` to be broken
    to multple lines as Python does not have inline comments.
    """
    return _TrailingCommentedValue(value, comment_str)


def unwrap_comments(value):
    comment = None
    trailing_comment = None

    while isinstance(value, (_CommentedValue, _TrailingCommentedValue)):
        if isinstance(value, _CommentedValue):
            comment = value.comment
            value = value.value
        elif isinstance(value, _TrailingCommentedValue):
            trailing_comment = value.comment
            value = value.value

    return (value, comment, trailing_comment)


def is_commented(value):
    return (
        isinstance(value, Annotated) and
        isinstance(value.annotation, CommentAnnotation)
    )


def builtin_identifier(s):
    return annotate(Token.NAME_BUILTIN, s)


def identifier(s):
    return annotate(Token.NAME_FUNCTION, s)


def general_identifier(s):
    if callable(s):
        module, qualname = s.__module__, s.__qualname__

        if module in IMPLICIT_MODULES:
            if module == 'builtins':
                return builtin_identifier(qualname)
            return identifier(qualname)
        return identifier(f'{module}.{qualname}')
    return identifier(s)


def classattr(cls, attrname):
    return concat([
        general_identifier(cls),
        identifier(f'.{attrname}')
    ])


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


def _run_pretty(pretty_fn, value, ctx, trailing_comment=None):
    if ctx.is_visited(value):
        return _pretty_recursion(value)

    ctx.start_visit(value)

    if trailing_comment:
        doc = pretty_fn(value, ctx, trailing_comment=trailing_comment)
    else:
        doc = pretty_fn(value, ctx)

    if not (
        isinstance(doc, str) or
        isinstance(doc, Doc)
    ):
        fnname = f'{pretty_fn.__module__}.{pretty_fn.__qualname__}'
        raise ValueError(
            'Functions decorated with register_pretty must return '
            f'an instance of str or Doc. {fnname} returned '
            f'{repr(doc)} instead.'
        )

    ctx.end_visit(value)

    return doc


_PREDICATE_REGISTRY = []


def _repr_pretty(value, ctx):
    for predicate, fn in _PREDICATE_REGISTRY:
        if predicate(value):
            return fn(value, ctx)
    return repr(value)


pretty_dispatch = singledispatch(partial(_run_pretty, _repr_pretty))


def pretty_python_value(value, ctx):
    comment = None
    trailing_comment = None

    value, comment, trailing_comment = unwrap_comments(value)

    if trailing_comment:
        doc = pretty_dispatch(
            value,
            ctx,
            trailing_comment=trailing_comment
        )
    else:
        doc = pretty_dispatch(
            value,
            ctx
        )

    if comment:
        return annotate_comment(
            comment,
            doc
        )
    return doc


def register_pretty(type=None, predicate=None):
    """Returns a decorator that registers the decorated function
    as the pretty printer for instances of ``type``.
    """

    if type is None and predicate is None:
        raise ValueError(
            "You must provide either the 'type' or 'predicate' argument."
        )

    if type is not None and predicate is not None:
        raise ValueError(
            "You must provide either the 'type' or 'predicate' argument,"
            "but not both"
        )

    if predicate is not None:
        if not callable(predicate):
            raise ValueError(
                f"Expected a callable for 'predicate', got {repr(predicate)}"
            )

    def decorator(fn):
        sig = inspect.signature(fn)

        value = None
        ctx = None

        try:
            sig.bind(value, ctx)
        except TypeError:
            fnname = f'{fn.__module__}.{fn.__qualname__}'
            raise ValueError(
                "Functions decorated with register_pretty must accept "
                "exactly two positional parameters: 'value' and 'ctx'. "
                f"The function signature for {fnname} was not compatible."
            )

        if type:
            pretty_dispatch.register(type, partial(_run_pretty, fn))
        else:
            assert callable(predicate)
            _PREDICATE_REGISTRY.append((predicate, fn))
        return fn
    return decorator


def bracket(ctx, left, child, right):
    return concat([
        left,
        nest(ctx.indent, concat([SOFTLINE, child])),
        SOFTLINE,
        right
    ])


def commentdoc(text):
    """Returns a Doc representing a comment `text`. `text` is
    treated as words, and any whitespace may be used to break
    the comment to multiple lines."""
    if not text:
        raise ValueError(
            f'Expected non-empty comment str, got {repr(text)}'
        )

    commentlines = []
    for line in text.splitlines():
        alternating_words_ws = list(filter(None, WHITESPACE_PATTERN_TEXT.split(line)))
        starts_with_whitespace = bool(
            WHITESPACE_PATTERN_TEXT.match(alternating_words_ws[0])
        )

        if starts_with_whitespace:
            prefix = alternating_words_ws[0]
            alternating_words_ws = alternating_words_ws[1:]
        else:
            prefix = NIL

        if len(alternating_words_ws) % 2 == 0:
            # The last part must be whitespace.
            alternating_words_ws = alternating_words_ws[:-1]

        for idx, tup in enumerate(zip(alternating_words_ws, cycle([False, True]))):
            part, is_ws = tup
            if is_ws:
                alternating_words_ws[idx] = flat_choice(
                    when_flat=part,
                    when_broken=always_break(
                        concat([
                            HARDLINE,
                            '# ',
                        ])
                    )
                )

        commentlines.append(
            concat([
                '# ',
                prefix,
                fill(alternating_words_ws)
            ])
        )

    outer = identity

    if len(commentlines) > 1:
        outer = always_break

    return annotate(
        Token.COMMENT_SINGLE,
        outer(concat(intersperse(HARDLINE, commentlines)))
    )


def sequence_of_docs(ctx, left, docs, right, dangle=False):
    docs = list(docs)

    # Performance optimization:
    # in case of really long sequences,
    # the layout algorithm can be quite slow.
    # No branching here is needed if the sequence
    # is long enough that even with the shortest
    # element output, it does not fit the ribbon width.
    minimum_output_len = (
        2 +  # Assume left and right are one character each
        len(', ') * (len(docs) - 1) +
        len(docs)  # each element must take at least one character
    )

    MAX_PRACTICAL_RIBBON_WIDTH = 150

    will_break = minimum_output_len > MAX_PRACTICAL_RIBBON_WIDTH

    has_comment = any(is_commented(doc) for doc in docs)

    parts = []
    for idx, doc in enumerate(docs):
        last = idx == len(docs) - 1

        if is_commented(doc):
            comment_str = doc.annotation.value
            # Try to fit the comment at the end of the same line.
            flat_version = concat([
                doc,
                COMMA if not last else NIL,
                '  ',
                commentdoc(comment_str),
                HARDLINE if not last else NIL
            ])

            # If the value is broken to multiple lines, add
            # comment on the line above.
            broken_version = concat([
                commentdoc(comment_str),
                HARDLINE,
                doc,
                COMMA if not last else NIL,
                HARDLINE if not last else NIL
            ])
            parts.append(
                group(
                    flat_choice(
                        when_flat=flat_version,
                        when_broken=broken_version,
                    )
                )
            )
        else:
            parts.append(doc)
            if not last:
                parts.append(
                    concat([COMMA, LINE])
                )

    if dangle:
        parts.append(COMMA)

    outer = (
        always_break
        if will_break or has_comment
        else group
    )

    return outer(bracket(ctx, left, concat(parts), right))


def prettycall(ctx, fn, *args, **kwargs):
    """Returns a Doc that represents a function call to :keyword:`fn` with
    the remaining positional and keyword arguments.

    Given an arbitrary context ``ctx``,::

        prettycall(ctx, sorted, [7, 4, 5], reverse=True)

    Will result in output::

        sorted([7, 4, 5], reverse=True)

    The layout algorithm will automatically break the call to multiple
    lines if needed::

        sorted(
            [7, 4, 5],
            reverse=True
        )

    ``prettycall`` automatically handles syntax highlighting.

    :param ctx: a context value
    :type ctx: peprint.peprint.PrettyContext
    :param fn: a callable
    :param args: positional arguments to render to the call
    :param kwargs: keyword arguments to render to the call
    :returns: :class:`~peprint.doc.Doc`
    """

    fndoc = general_identifier(fn)

    if ctx.depth_left <= 0:
        return concat([fndoc, LPAREN, ELLIPSIS, RPAREN])

    if not kwargs and len(args) == 1:
        sole_arg = args[0]
        unwrapped_sole_arg, _comment, _trailing_comment = unwrap_comments(args[0])
        if type(unwrapped_sole_arg) in (list, dict, tuple):
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
    hug_sole_arg=False,
    trailing_comment=None,
):
    """Builds a doc that looks like a function call,
    from docs that represent the function, arguments
    and keyword arguments.

    If ``hug_sole_arg`` is True, and the represented
    functional call is done with a single non-keyword
    argument, the function call parentheses will hug
    the sole argument doc without newlines and indentation
    in break mode. This makes a difference in calls
    like this::

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

    If ``trailing_comment`` is provided, the text is
    rendered as a comment after the last argument and
    before the closing parenthesis. This will force
    the function call to be broken to multiple lines.
    """
    if callable(fndoc):
        fndoc = general_identifier(fndoc)

    has_comment = bool(trailing_comment)

    argdocs = list(argdocs)
    kwargdocs = list(kwargdocs)

    kwargdocs = [
        # Propagate any comments to the kwarg doc.
        (
            annotate_comment(
                doc.annotation.value,
                concat([binding, ASSIGN_OP, doc.doc])
            )
            if is_commented(doc)
            else concat([binding, ASSIGN_OP, doc])
        )
        for binding, doc in kwargdocs
    ]

    if not (argdocs or kwargdocs):
        return concat([
            fndoc,
            LPAREN,
            RPAREN,
        ])

    if (
        hug_sole_arg and
        not kwargdocs and
        len(argdocs) == 1 and
        not is_commented(argdocs[0])
    ):
        return group(
            concat([
                fndoc,
                LPAREN,
                argdocs[0],
                RPAREN
            ])
        )

    allarg_docs = [*argdocs, *kwargdocs]

    if trailing_comment:
        allarg_docs.append(commentdoc(trailing_comment))

    parts = []

    for idx, doc in enumerate(allarg_docs):
        last = idx == len(allarg_docs) - 1

        if is_commented(doc):
            has_comment = True
            comment_str = doc.annotation.value
            doc = doc.doc
        else:
            comment_str = None

        part = concat([doc, NIL if last else COMMA])

        if comment_str:
            part = group(
                flat_choice(
                    when_flat=concat([
                        part,
                        '  ',
                        commentdoc(comment_str)
                    ]),
                    when_broken=concat([
                        commentdoc(comment_str),
                        HARDLINE,
                        part,
                    ]),
                )
            )

        if not last:
            part = concat([part, HARDLINE if has_comment else LINE])

        parts.append(part)

    outer = (
        always_break
        if has_comment
        else group
    )

    return outer(
        concat([
            fndoc,
            LPAREN,
            nest(
                ctx.indent,
                concat([
                    SOFTLINE,
                    concat(parts),
                ])
            ),
            SOFTLINE,
            RPAREN
        ])
    )


@register_pretty(tuple)
@register_pretty(list)
@register_pretty(set)
def pretty_bracketable_iterable(value, ctx, trailing_comment=None):
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

    if trailing_comment:
        els = chain(els, [commentdoc(trailing_comment)])
        dangle = False

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

    has_comment = False

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

        kcomment = None
        if is_commented(kdoc):
            has_comment = True
            kcomment = kdoc.annotation.value
            kdoc = kdoc.doc

        vcomment = None
        if is_commented(vdoc):
            has_comment = True
            vcomment = vdoc.annotation.value
            vdoc = vdoc.doc

        pairs.append((k, v, kdoc, vdoc, kcomment, vcomment))

    parts = []
    for idx, tup in enumerate(pairs):
        last = idx == len(pairs) - 1

        k, v, kdoc, vdoc, kcomment, vcomment = tup

        if not (kcomment or vcomment):
            parts.append(
                concat([
                    kdoc,
                    concat([COLON, ' ']),
                    vdoc,
                    NIL if last else COMMA,
                    NIL if last else LINE,
                ]),
            )
            continue

        if kcomment:
            kcommented = concat([
                commentdoc(kcomment),
                HARDLINE,
                kdoc,
            ])
        else:
            kcommented = kdoc

        if vcomment:
            vcommented = group(
                flat_choice(
                    # Add comment at the end of the line
                    when_flat=concat([
                        vdoc,
                        NIL if last else COMMA,
                        '  ',
                        commentdoc(vcomment),
                        NIL if last else HARDLINE,
                    ]),

                    # Put comment above the value
                    # on its own line
                    when_broken=concat([
                        nest(
                            ctx.indent,
                            concat([
                                HARDLINE,
                                commentdoc(vcomment),
                                HARDLINE,
                                # Rerender vdoc with plain multiline strategy,
                                # since we already have an indentation.
                                pretty_python_value(
                                    v,
                                    ctx=(
                                        ctx
                                        .nested_call()
                                        .use_multiline_strategy(MULTILINE_STATEGY_PLAIN)
                                    ),
                                ),
                                COMMA if not last else NIL,
                                HARDLINE if not last else NIL
                            ])
                        ),
                    ])
                )
            )
        else:
            vcommented = concat([
                vdoc,
                COMMA if not last else NIL,
                LINE if not last else NIL
            ])

        parts.append(
            concat([
                kcommented,
                concat([COLON, ' ']),
                vcommented
            ])
        )

    doc = bracket(
        ctx,
        LBRACE,
        concat(parts),
        RBRACE,
    )

    if len(pairs) > 2 or has_comment:
        doc = always_break(doc)
    else:
        doc = group(doc)

    return doc


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


STR_LITERAL_ESCAPES = re.compile(
    r'''((?:\\[\\abfnrtv"'])|'''
    r'(?:\\N\{.*?\})|'
    r'(?:\\u[a-fA-F0-9]{4})|'
    r'(?:\\U[a-fA-F0-9]{8})|'
    r'(?:\\x[a-fA-F0-9]{2})|'
    r'(?:\\[0-7]{1,3}))'
)


def highlight_escapes(s):
    if not s:
        return NIL

    matches = STR_LITERAL_ESCAPES.split(s)

    starts_with_match = bool(STR_LITERAL_ESCAPES.match(matches[0]))

    docs = []
    for part, is_escaped in zip(
        matches,
        cycle([starts_with_match, not starts_with_match])
    ):
        if not part:
            continue

        docs.append(
            annotate(
                (
                    Token.STRING_ESCAPE
                    if is_escaped
                    else Token.LITERAL_STRING
                ),
                part
            )
        )

    return concat(docs)


def pretty_single_line_str(s, indent, use_quote=None):
    prefix = (
        annotate(Token.STRING_AFFIX, 'b')
        if isinstance(s, bytes)
        else ''
    )

    if use_quote is None:
        use_quote = determine_quote_strategy(s)

    escaped = escape_str_for_quote(use_quote, s)
    escapes_highlighted = highlight_escapes(escaped)

    return concat([
        prefix,
        annotate(
            Token.LITERAL_STRING,
            concat([
                use_quote,
                escapes_highlighted,
                use_quote
            ])
        )
    ])


def split_at(idx, sequence):
    return (sequence[:idx], sequence[idx:])


def escaped_len(s, use_quote):
    return len(escape_str_for_quote(use_quote, s))


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
    curr_line_len = 0
    while remaining_stack:
        curr, is_whitespace = remaining_stack.pop()
        curr_line_parts.append(curr)
        curr_line_len += escaped_len(curr, use_quote)

        if curr_line_len == max_len:
            if not is_whitespace and len(curr_line_parts) > 2:
                curr_line_parts.pop()
                yield empty.join(curr_line_parts)
                curr_line_parts = []
                curr_line_len = 0
                remaining_stack.append((curr, is_whitespace))
            else:
                yield empty.join(curr_line_parts)
                curr_line_parts = []
                curr_line_len = 0
                continue
        elif curr_line_len > max_len:
            if not is_whitespace and len(curr_line_parts) > 1:
                curr_line_parts.pop()
                yield empty.join(curr_line_parts)
                remaining_stack.append((curr, is_whitespace))
                curr_line_parts = []
                curr_line_len = 0
                continue

            curr_line_parts.pop()

            remaining_len = max_len - (curr_line_len - escaped_len(curr, use_quote))
            this_line_part, next_line_part = split_at(max(remaining_len, 0), curr)

            curr_line_parts.append(this_line_part)

            yield empty.join(curr_line_parts)
            curr_line_parts = []
            curr_line_len = 0

            if next_line_part:
                remaining_stack.append((next_line_part, is_whitespace))

    if curr_line_parts:
        yield empty.join(curr_line_parts)


@register_pretty(str)
@register_pretty(bytes)
def pretty_str(s, ctx):
    # Subclasses of str/bytes
    # will be printed as StrSubclass('the actual string')
    constructor = type(s)
    is_native_type = constructor in (str, bytes)

    if ctx.depth_left == 0:
        if isinstance(s, str):
            return prettycall(ctx, constructor, ...)
        else:
            assert isinstance(s, bytes)
            return prettycall(ctx, constructor, ...)

    multiline_strategy = ctx.multiline_strategy
    peprint_indent = ctx.indent

    def evaluator(indent, column, page_width, ribbon_width):
        nonlocal multiline_strategy

        columns_left_in_line = page_width - column
        columns_left_in_ribbon = indent + ribbon_width - column
        available_width = min(columns_left_in_line, columns_left_in_ribbon)

        singleline_str_chars = len(s) + len('""')
        flat_version = pretty_single_line_str(s, peprint_indent)

        if singleline_str_chars <= available_width:
            if is_native_type:
                return flat_version
            return build_fncall(ctx, constructor, argdocs=[flat_version])

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

        if not is_native_type:
            multiline_strategy = MULTILINE_STATEGY_PLAIN

        if multiline_strategy == MULTILINE_STATEGY_PLAIN:
            res = always_break(concat(parts))
            if is_native_type:
                return res
            return build_fncall(ctx, constructor, argdocs=[res])
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


def _pretty_recursion(value):
    return f'<Recursion on {type(value).__name__} with id={id(value)}>'


def python_to_sdocs(value, indent, width, depth, ribbon_width=71):
    if depth is None:
        depth = float('inf')

    doc = pretty_python_value(
        value,
        ctx=PrettyContext(indent=indent, depth_left=depth, visited=set())
    )

    if is_commented(doc):
        doc = group(
            flat_choice(
                when_flat=concat([
                    doc,
                    '  ',
                    commentdoc(doc.annotation.value),
                ]),
                when_broken=concat([
                    commentdoc(doc.annotation.value),
                    HARDLINE,
                    doc
                ])
            )
        )

    ribbon_frac = min(1.0, ribbon_width / width)

    return layout_smart(doc, width=width, ribbon_frac=ribbon_frac)
