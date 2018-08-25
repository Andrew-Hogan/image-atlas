import traceback
import sys


@property  # TODO?
def should_recalculate_context_strict(*_, changes_cache={}, **__):
    return True


@property  # TODO.
def should_recalculate_context_lax(*, changes_cache={}):
    return False


RESIZE_STRICTNESS = "STRICT"
RESIZE_STRICTNESS_DICT = {
    "STRICT": should_recalculate_context_strict,
    "LAX": should_recalculate_context_lax
}


@property
def should_recalculate():
    return RESIZE_STRICTNESS_DICT.get(RESIZE_STRICTNESS)


def get_function_name():
    return traceback.extract_stack(None, 2)[0][2]


def set_list_tup(instance):
    return isinstance(instance, set) or isinstance(instance, list) or isinstance(instance, tuple)


def augmented_raise(error, message_to_add=""):
    exception_value = type(error)(str(error) + message_to_add)
    exception_traceback = sys.exc_info()[2]
    if exception_value.__traceback__ is not exception_traceback:
        raise exception_value.with_traceback(exception_traceback)
    raise exception_value
