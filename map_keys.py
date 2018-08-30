import functools
from types import FunctionType


_ACTIVE_CLASS = "_active_descriptor"
_PROTECTED_SELF = "_protected_self"
_SHOULD_NOT_DECORATE_FLAG = "_protect_self_reference"
DEFAULT_NO_DECORATES_ON_ANY_IN_INHERITANCE_TREE = {
    "__new__", "__init__", "__getattr__", "__delattr__", "__getattribute__", "__del__", "queen", "drone", "__setattr__",
    "__dict__", "__str__", "__repr__", "__set__", "__eq__", "__hash__", "__init__", "__new__"
}
DEFAULT_FORCED_DECORATES_ON_DECORATED_CLASS_ONLY = {
    "__eq__", "__hash__", "__init__", "__new__", "__setattr__"
}


def resist(this_function):
    this_function._protect_self_reference = False
    return this_function


def _should_protect_self_access(attr, value):
    return (
        attr not in DEFAULT_NO_DECORATES_ON_ANY_IN_INHERITANCE_TREE and isinstance(value, FunctionType)
        and getattr(value, "_protect_self_reference", True)
    )


def _safe_self_access_decorator(wrapped_method):
    @functools.wraps(wrapped_method)
    def method_wrapper(self, *args, **kwargs):
        if hasattr(self, "queen"):
            return wrapped_method(self.queen, *args, **kwargs)
        return wrapped_method(self, *args, **kwargs)
    return method_wrapper


def assimilate(_wrapped_class=None, *, default_class=None, _directly_wrapped_classes=set(), _from_sub_canary=[]):
    """

    Assumptions:
        1. That any of the attributes set by this wrapper (self.queen, self.drone, _protected_self, _active_descriptor,
            and _protect_self_reference [the last is on methods]) shall not be changed except through the auto-interface
            (@assimilate, @resist) offered by this toolkit.
        2. That any class to be decorated will not have its subclasses @assimilate decorated as well, as they will be
            auto-decorated.

    :param _wrapped_class:
    :param default_class:
    :return:
    """
    if default_class is None:
        default_class = ShapeDescriptor

    def borg_pod_decorator(wrapped_class):
        def _setup_pod_in_new(wrapped_new):
            @functools.wraps(wrapped_new)
            def new_wrapper(cls, *args, queen=None, _base_class=None, **kwargs):
                if _from_sub_canary:
                    print("FROM SUB NEW CALLED")
                    new = wrapped_new(cls)
                    return [new]
                print("NEW CALLED")
                if _base_class is None:
                    _base_class = default_class
                is_wrapped_subclass = True
                if not any((pappy in _directly_wrapped_classes for pappy in ancestors)):
                    if _should_be_self_unless_multi_wrapped == cls:
                        is_wrapped_subclass = False
                if queen is None:
                    for ids, arg in enumerate(args):
                        if isinstance(arg, _base_class):
                            queen = arg
                            args = args[:ids] + args[ids + 1:]
                            break
                    else:
                        queen = _base_class({})

                shared_state = queen.__dict__
                if is_wrapped_subclass:
                    _from_sub_canary.append(None)
                    new_object = wrapped_new(cls).pop()
                    del _from_sub_canary[-1]
                    print("FROM SUB NEW RECEIVED")
                else:
                    new_object = wrapped_new(cls)
                new_object.__init__(shared_state, *args, **kwargs)
                if is_wrapped_subclass:
                    del _from_sub_canary[-1]
                    print("SUB INIT FINISHED")
                return queen
            return new_wrapper

        def _assimilate_in_init(wrapped_init):
            @functools.wraps(wrapped_init)
            def init_wrapper(self, shared_state, *args, **kwargs):
                if any((pappy in _directly_wrapped_classes for pappy in ancestors)):
                    if _from_sub_canary:
                        return wrapped_init(self, shared_state, *args, **kwargs)
                    else:
                        _from_sub_canary.append(None)
                elif _from_sub_canary:
                    return ancestors[0].__init__(self, *args, **kwargs)
                self.__dict__ = shared_state
                self._active_descriptor = self
                self.queen = self._protected_self.queen
                self.drone = self._protected_self.drone
                if _from_sub_canary:
                    return wrapped_init(self, shared_state, *args, **kwargs)
                return wrapped_init(self, *args, **kwargs)
            return init_wrapper

        def _modify_methods_for_self_reference(this_class):
            for c_attribute, c_method in this_class.__dict__.copy().items():
                if _should_protect_self_access(c_attribute, c_method):
                    print("{} wrapped by method dec.".format(c_attribute))
                    setattr(this_class, c_attribute, _safe_self_access_decorator(this_class.__dict__[c_attribute]))
                    c_method._protect_self_reference = False

        def _borg_pod_set_with_safe_self_access(wrapped_method):
            @functools.wraps(wrapped_method)
            def setter_wrapper(self, attribute, value):
                if _should_protect_self_access(attribute, value):
                    print("WRAPPING IT: {}".format(attribute))
                    value = _safe_self_access_decorator(value)
                for super_class in _all_ancestors:
                    modified_setter = hasattr(super(super_class, self).__setattr__, "_protect_self_reference")
                    if not modified_setter:
                        super(super_class, self).__setattr__(attribute, value)
                        break
            return setter_wrapper

        _directly_wrapped_classes.update({wrapped_class})
        _all_ancestors = wrapped_class.mro()
        _should_be_self_unless_multi_wrapped, ancestors = _all_ancestors[0], _all_ancestors[1:]
        print("WRAPPED CLASSES: {}".format(_directly_wrapped_classes))
        print(ancestors)

        # Instance method self-reference-return protector
        _modify_methods_for_self_reference(wrapped_class)

        setattr(wrapped_class, '__new__', _setup_pod_in_new(wrapped_class.__new__))  # Check for pre-set?
        wrapped_class.__new__._protect_self_reference = False
        setattr(wrapped_class, '__init__', _assimilate_in_init(wrapped_class.__init__))
        wrapped_class.__init__._protect_self_reference = False
        setattr(wrapped_class, '__hash__', lambda x: hash(x.queen))
        wrapped_class.__hash__._protect_self_reference = False
        setattr(wrapped_class, '__eq__', lambda x, y: x.queen is y.queen if hasattr(y, "queen") else False)
        wrapped_class.__eq__._protect_self_reference = False
        setattr(wrapped_class, '__setattr__', _borg_pod_set_with_safe_self_access(wrapped_class.__setattr__))
        wrapped_class.__setattr__._protect_self_reference = False

        for ancestor in ancestors:
            print(ancestor)
            if ancestor is not object:
                _modify_methods_for_self_reference(ancestor)

        return wrapped_class

    if _wrapped_class is None:
        return borg_pod_decorator
    return borg_pod_decorator(_wrapped_class)


class ShapeDescriptor(object):
    _base_borgs = set()

    def __init__(self, _shared_state=None):
        self.__dict__ = _shared_state if _shared_state is not None else {}
        self._active_descriptor = self
        if _PROTECTED_SELF not in self.__dict__:
            self._protected_self = self

    @property
    def queen(self):
        return self._protected_self

    @property
    def drone(self):
        return self._active_descriptor

    def __getattr__(self, name):
        if _ACTIVE_CLASS in self.__dict__:
            active_class = self.__dict__[_ACTIVE_CLASS]
            if active_class is self.__dict__[_PROTECTED_SELF]:
                raise AttributeError("Base borg-pod does not have attribute {}.".format(name))
            if hasattr(active_class, name):
                return getattr(active_class, name)
        raise AttributeError("Base borg-pod does not have attribute {}.".format(name))

    def __str__(self):
        if self._protected_self is not self._active_descriptor:
            return self._active_descriptor.__str__()
        return "<Unbound {} object #{}>".format(self._protected_self.__class__.__name__, id(self._protected_self))

    __repr__ = __str__

    def __hash__(self):
        return hash((self.__class__.__name__, id(self)))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Useless(object):
    def __init__(self, *args, **kwargs):
        print(args)
        print(kwargs)


class PerfectGreekInfluencedChalkDrawingOfFace(Useless):
    def __init__(self, *args, **kwargs):
        print(self.__class__)
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "pre-circle"

    def self_method(self):
        return self


print("C_SPONGE")
print(PerfectGreekInfluencedChalkDrawingOfFace)
print(type(PerfectGreekInfluencedChalkDrawingOfFace))


@assimilate(default_class=ShapeDescriptor)
class Circle(PerfectGreekInfluencedChalkDrawingOfFace):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "circle"

    @staticmethod
    def info():
        print("I AM CIRCLE.")

    # def self_method(self):
    #     return self

    def __str__(self):
        if hasattr(self, "drone"):
            return "<{} object #{} linked to {} #{}>".format(
                self.drone.__class__.__name__, id(self.drone), self.queen.__class__.__name__, id(self.queen)
            )
        return "<Unassimilated {} object #{}>".format(self.__class__.__name__, id(self))

    __repr__ = __str__


print("C")
print(Circle)
print(type(Circle))


@assimilate
class Ellipse(Circle):
    def __init__(self, *args, **kwargs):
        print("ELLIPSE CLASS INIT CALLED")
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "ellipse"


print("C-sponge")
print(Ellipse)
print(type(Ellipse))


@assimilate
class AlphaNumeric(object):
    def __init__(self):
        self.shape_type = "character"

    @staticmethod
    def info():
        print("I AM CHARACTER.")

    def self_method(self):
        self.info()
        return self

    def __str__(self):
        if hasattr(self, "drone"):
            return "<{} object #{} linked to {} #{}>".format(
                self.drone.__class__.__name__, id(self.drone), self.queen.__class__.__name__, id(self.queen)
            )
        return "<Unassimilated {} object #{}>".format(self.__class__.__name__, id(self))

    __repr__ = __str__


print("C-Alpha")
print(AlphaNumeric)
print(type(AlphaNumeric))


@assimilate
class Punctuation(object):
    def __init__(self):
        self.shape_type = "punctuation"

    @staticmethod
    def info():
        print("I AM PUNCTUATION.")

    @resist
    def self_method(self):
        return self

    def __str__(self):
        if hasattr(self, "drone"):
            return "<{} object #{} linked to {} #{}>".format(
                self.drone.__class__.__name__, id(self.drone), self.queen.__class__.__name__, id(self.queen)
            )
        return "<Unassimilated {} object #{}>".format(self.__class__.__name__, id(self))

    __repr__ = __str__


print("C-Punct")
print(Punctuation)
print(type(Punctuation))


def compare_seq(sequence, sequence_2=None):
    if sequence_2 is None:
        for ob_a, ob_b in zip(sequence, sequence[1::] + [sequence[0]]):
            print("{}: {} is {}".format(ob_a is ob_b, ob_a, ob_b))
    else:
        for ob_a, ob_b in zip(sequence, sequence_2):
            print("{}: {} is {}".format(ob_a is ob_b, ob_a, ob_b))


def assert_seq(sequence, sequence_2=None, *, assert_val=True):
    if sequence_2 is None:
        for ob_a, ob_b in zip(sequence, sequence[1::] + [sequence[0]]):
            assert (ob_a is ob_b) == assert_val, "Assertion that {} is {} did not match provided value of {}.".format(
                ob_a, ob_b, assert_val
            )
    else:
        for ob_a, ob_b in zip(sequence, sequence_2):
            assert (ob_a is ob_b) == assert_val, "Assertion that {} is {} did not match provided value of {}.".format(
                ob_a, ob_b, assert_val
            )


def convert_seq(sequence, new_class):
    return [new_class(obj) for obj in sequence]


def printable_comparisons_demo(num_objects=6):
    test_objects_original = [ShapeDescriptor() for _ in range(num_objects)]
    print("Is equal to all?")
    compare_seq(test_objects_original)

    print("\nTo Circle-")
    test_objects_circle = convert_seq(test_objects_original, Circle)
    print("Are they unique objects?")
    compare_seq(test_objects_circle)
    print("Is equal to old version?")
    compare_seq(test_objects_circle, test_objects_original)

    print("\nTo Characters-")
    test_objects_characters = convert_seq(test_objects_circle, AlphaNumeric)
    print("Are they unique objects?")
    compare_seq(test_objects_characters)
    print("Is equal to circle list?")
    compare_seq(test_objects_characters, test_objects_circle)
    print("Is equal to old version?")
    compare_seq(test_objects_characters, test_objects_original)

    print("\n____\nWhat if we return self from a method?\n")
    self_list_protected = [obj.self_method() for obj in test_objects_characters]
    print("'self' is automatically converted to the queen for consistency!")
    print("Do they all evaluate as the same?")
    compare_seq(self_list_protected)
    print("Does it still evaluate as the same as the previous characters list?")
    compare_seq(self_list_protected, test_objects_characters)
    print("\nLet's try converting to a class with a @resist decorated method returning 'self'.\n____\n")

    print("To Punctuation-")
    test_objects_punctuation = convert_seq(test_objects_characters, Punctuation)
    print("Are they unique objects?")
    compare_seq(test_objects_punctuation)
    print("Is equal to character list?")
    compare_seq(test_objects_punctuation, test_objects_characters)
    print("Is equal to circle list?")
    compare_seq(test_objects_punctuation, test_objects_circle)
    print("Is equal to old version?")
    compare_seq(test_objects_punctuation, test_objects_original)

    print("\n____\nWhat if we return self from a method decorated with @resist?\n")
    self_list_unprotected = [obj.self_method() for obj in test_objects_punctuation]
    print("A method decorated with @resist will not convert 'self' to the queen reference. (not suggested)")
    print("Are they unique objects?")
    compare_seq(self_list_unprotected)
    print("Does it still evaluate as the same as the previous punctuation list?")
    compare_seq(self_list_unprotected, test_objects_punctuation)
    print("\nWhat if we retrieve the drones from the previous characters list?")
    drone_list_characters = [obj.drone for obj in test_objects_characters]
    print("Does it evaluate as the same to the @resist self list?")
    compare_seq(drone_list_characters, self_list_unprotected)
    print("\nWhat if we retrieve the queen from the @resist self list?")
    self_list_restored = [obj.queen for obj in self_list_unprotected]
    print("Does it evaluate as the same to the original characters list?")
    compare_seq(self_list_restored, test_objects_characters)
    print("\nTests Complete\n____")


def main(num_objects=6):
    test_objects_original = [ShapeDescriptor() for _ in range(num_objects)]
    print("Are they unique objects?")
    assert_seq(test_objects_original, assert_val=False)

    print("\nTo Circle-")
    test_objects_circle = convert_seq(test_objects_original, Circle)
    print("Are they unique objects?")
    assert_seq(test_objects_circle, assert_val=False)
    print("Is equal to old version?")
    assert_seq(test_objects_circle, test_objects_original)
    print("what if we return self?")
    self_list_ambig = [obj.self_method() for obj in test_objects_circle]
    print(self_list_ambig)
    print("Are they equal to the old version?")
    compare_seq(self_list_ambig, test_objects_circle)
    print("What if we use instances of a parent class?")
    test_objects_undecorated_parent_class = [PerfectGreekInfluencedChalkDrawingOfFace() for _ in range(num_objects)]
    self_list_face = [obj.self_method() for obj in test_objects_undecorated_parent_class]
    print(self_list_face)
    print("What if we use a child class?")
    test_objects_decorated_subclass = [Ellipse() for _ in range(num_objects)]
    self_list_sub = [obj.self_method() for obj in test_objects_decorated_subclass]
    print(self_list_sub)

    print("\nTo Characters-")
    test_objects_characters = convert_seq(test_objects_circle, AlphaNumeric)
    print("Are they unique objects?")
    assert_seq(test_objects_characters, assert_val=False)
    print("Is equal to circle list?")
    assert_seq(test_objects_characters, test_objects_circle)
    print("Is equal to old version?")
    assert_seq(test_objects_characters, test_objects_original)

    print("\n____\nWhat if we return self from a method?\n")
    self_list_protected = [obj.self_method() for obj in test_objects_characters]
    print("'self' is automatically converted to the queen for consistency!")
    print("Are they unique objects?")
    assert_seq(self_list_protected, assert_val=False)
    print("Does it still evaluate as the same as the previous characters list?")
    assert_seq(self_list_protected, test_objects_characters)
    print("\nLet's try converting to a class with a @resist decorated method returning 'self'.\n____\n")

    print("To Punctuation-")
    test_objects_punctuation = convert_seq(test_objects_characters, Punctuation)
    print("Are they unique objects?")
    assert_seq(test_objects_punctuation, assert_val=False)
    print("Is equal to character list?")
    assert_seq(test_objects_punctuation, test_objects_characters)
    print("Is equal to circle list?")
    assert_seq(test_objects_punctuation, test_objects_circle)
    print("Is equal to old version?")
    assert_seq(test_objects_punctuation, test_objects_original)

    print("\n____\nWhat if we return self from a method decorated with @resist?\n")
    self_list_unprotected = [obj.self_method() for obj in test_objects_punctuation]
    print("A method decorated with @resist will not convert 'self' to the queen reference. (not suggested)")
    print("Are they unique objects?")
    assert_seq(self_list_unprotected, assert_val=False)
    print("Do they no longer evaluate as the same object from the previous punctuation list?")
    assert_seq(self_list_unprotected, test_objects_punctuation, assert_val=False)
    print("\nWhat if we retrieve the drones from the previous characters list?")
    drone_list_characters = [obj.drone for obj in test_objects_characters]
    print("Does it evaluate as the same to the @resist self list?")
    assert_seq(drone_list_characters, self_list_unprotected)
    print("\nWhat if we retrieve the queen from the @resist self list?")
    self_list_restored = [obj.queen for obj in self_list_unprotected]
    print("Does it evaluate as the same to the original characters list?")
    assert_seq(self_list_restored, test_objects_characters)
    print("\nTests Complete\n____")


if __name__ == "__main__":
    # printable_comparisons_demo()
    main()
