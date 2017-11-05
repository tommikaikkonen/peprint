from enum import Enum, unique, auto

from django.db.models.fields import NOT_PROVIDED
from django.db.models import Model, ForeignKey
from django.db.models.query import QuerySet

from ..peprint import (
    MULTILINE_STATEGY_HANG,
    LBRACKET,
    RBRACKET,
    build_fncall,
    comment,
    general_identifier,
    prettycall,
    pretty_python_value,
    register_pretty,
    sequence_of_docs,
)

from ..utils import find


QUERYSET_OUTPUT_SIZE = 20


@unique
class ModelVerbosity(Enum):
    UNSET = auto()
    MINIMAL = auto()
    SHORT = auto()
    FULL = auto()


def inc(value):
    return value


class dec(object):
    __slots__ = ('value', )

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        assert isinstance(other, dec)
        return self.value > other.value

    def __gt__(self, other):
        assert isinstance(other, dec)
        return self.value < other.value

    def __eq__(self, other):
        assert isinstance(other, dec)
        return self.value == other.value

    def __le__(self, other):
        assert isinstance(other, dec)
        return self.value >= other.value

    def __ge__(self, other):
        assert isinstance(other, dec)
        return self.value <= other.value

    __hash__ = None


def field_sort_key(field):
    return (
        dec(field.primary_key),
        dec(field.unique),
        inc(field.null),
        inc(field.blank),
        dec(field.default is NOT_PROVIDED),
        inc(field.name),
    )


def pretty_base_model(instance, ctx):
    verbosity = ctx.get(ModelVerbosity)

    model = type(instance)

    if verbosity == ModelVerbosity.MINIMAL:
        fields = [find(lambda field: field.primary_key, model._meta.fields)]
    elif verbosity == ModelVerbosity.SHORT:
        fields = sorted(
            (
                field
                for field in model._meta.fields
                if field.primary_key or not field.null and field.unique
            ),
            key=field_sort_key
        )
    else:
        fields = sorted(model._meta.fields, key=field_sort_key)

    attrs = []
    value_ctx = (
        ctx
        .nested_call()
        .use_multiline_strategy(MULTILINE_STATEGY_HANG)
    )

    for field in fields:
        if isinstance(field, ForeignKey):
            fk_value = getattr(instance, field.attname)
            if fk_value is not None:
                related_field = field.target_field
                related_model = related_field.model
                attrs.append((
                    field.name,
                    prettycall(
                        ctx,
                        related_model,
                        **{related_field.name: fk_value}
                    )
                ))
            else:
                attrs.append((field.name, pretty_python_value(None, value_ctx)))
        else:
            value = getattr(instance, field.name)

            if field.default is not NOT_PROVIDED:
                if callable(field.default):
                    default_value = field.default()
                else:
                    default_value = field.default

                if value == default_value:
                    continue

            attrs.append((field.name, pretty_python_value(value, value_ctx)))

    return build_fncall(
        ctx,
        model,
        kwargdocs=attrs
    )


def pretty_queryset(queryset, ctx):
    qs_cls = type(queryset)

    instances = list(queryset[:QUERYSET_OUTPUT_SIZE + 1])
    element_ctx = ctx.set(ModelVerbosity, ModelVerbosity.SHORT)
    docs = [
        pretty_python_value(value, element_ctx)
        for value in instances
    ]

    if len(instances) > QUERYSET_OUTPUT_SIZE:
        docs[-1] = comment('...remaining elements truncated')

    listdoc = sequence_of_docs(
        ctx,
        LBRACKET,
        docs,
        RBRACKET,
        dangle=False
    )

    return build_fncall(
        ctx,
        general_identifier(qs_cls),
        argdocs=[listdoc]
    )


def install():
    register_pretty(Model)(pretty_base_model)
    register_pretty(QuerySet)(pretty_queryset)
