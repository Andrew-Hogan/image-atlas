import functools
from types import FunctionType
import inspect


_ACTIVE_CLASS = "_active_descriptor"
_PROTECTED_SELF = "_protected_self"
_DEFAULT_NO_DECORATES = {"__new__", "__init__", "__getattr__", "__delattr__", "__getattribute__", "__del__",
                         "queen", "drone"}


def resist(f):
    f._protect_self_reference = False
    return f


def assimilate(_wrapped_class=None, *, default_class=None):
    if default_class is None:
        default_class = ShapeDescriptor

    def should_decorate(attr, value):
        return (attr not in _DEFAULT_NO_DECORATES
                and isinstance(value, FunctionType)
                and getattr(value, "_protect_self_reference", True))

    def method_decorator(wrapped_method):
        @functools.wraps(wrapped_method)
        def method_wrapper(self, *args, **kwargs):
            print("SELF: {}".format(self))
            ret = wrapped_method(self, *args, **kwargs)
            print("RET: {}".format(ret))
            if ret is self:
                print("SUBSTITUTING: {}".format(self.queen))
                return self.queen
            return ret
        return method_wrapper

    def borg_pod_safe_set(wrapped_method):
        @functools.wraps(wrapped_method)
        def setter_wrapper(self, attribute, value):
            if should_decorate(attribute, value):
                value = method_decorator(value)
            super(self.__class__, self).__setattr__(attribute, value)
        return setter_wrapper

    def borg_pod_decorator(wrapped_class):
        @functools.wraps(wrapped_class)
        def pod_wrapper(*c_args, **c_kwargs):
            def _setup_pod_in_new(wrapped_new):
                @functools.wraps(wrapped_new)
                def new_wrapper(cls, *args, queen=None, _base_class=default_class, **kwargs):
                    if queen is None:
                        for ids, arg in enumerate(args):
                            if isinstance(arg, _base_class):
                                queen = arg
                                args = args[:ids] + args[ids + 1:]
                                break
                        else:
                            queen = _base_class({})
                    print("args: {}".format(args))
                    print("kwargs: {}".format(kwargs))
                    shared_state = queen.__dict__
                    new_object = wrapped_new(cls)
                    new_object.__init__(shared_state, *args, **kwargs)
                    return queen
                return new_wrapper

            def _assimilate_in_init(wrapped_init):
                @functools.wraps(wrapped_init)
                def init_wrapper(self, shared_state, *args, **kwargs):
                    self.__dict__ = shared_state
                    self._active_descriptor = self
                    self.queen = self._protected_self.queen
                    self.drone = self._protected_self.drone
                    return wrapped_init(self, *args, **kwargs)
                return init_wrapper

            # Instance method self-reference-return protector
            for attribute, method in wrapped_class.__dict__.copy().items():
                if should_decorate(attribute, method):
                    setattr(wrapped_class, attribute, method_decorator(method))
            setattr(wrapped_class, '__setattr__', borg_pod_safe_set(wrapped_class.__setattr__))

            # __new__ self-reference-return protector & init setup
            wrapped_class.__new__ = _setup_pod_in_new(wrapped_class.__new__)
            wrapped_class.__init__ = _assimilate_in_init(wrapped_class.__init__)

            return wrapped_class(*c_args, **c_kwargs)
        return pod_wrapper

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
            if active_class == self.__dict__[_PROTECTED_SELF]:
                raise AttributeError("Base borg-pod does not have attribute {}.".format(name))
            if hasattr(active_class, name):
                return getattr(active_class, name)
        raise AttributeError("Base borg-pod does not have attribute {}.".format(name))


class ShapeDescriptorB(ShapeDescriptor):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)


@assimilate
class Circle(object):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "circle"
        print("CIRCLE INIT CALLED")

    @staticmethod
    def info():
        print("I AM CIRCLE.")

    def self_method(self):
        print(self.shape_type)
        return self


@assimilate
class AlphaNumeric(object):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "char"
        print("CHAR INIT CALLED")

    @staticmethod
    def info():
        print("I AM CHARACTER.")

    def self_method(self):
        print(self.shape_type)
        return self


@assimilate
class Punctuation(object):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.shape_type = "punct"
        print("PUNCT INIT CALLED")

    @staticmethod
    def info():
        print("I AM PUNCTUATION.")

    @resist
    def self_method(self):
        print(self.shape_type)
        return self


def collective_test():
    print("Begin test!")
    object_a = ShapeDescriptor()
    object_b = ShapeDescriptor()
    print("Is same object?")
    print(object_b is object_a)
    print("Translating to Circle.")
    object_c = Circle(object_b)
    print("Circle done.")
    print("New info:")
    object_c.info()
    # object_b.info()
    print("Is same object?")
    print(object_b is object_c)
    object_d = AlphaNumeric(object_c)
    print("New info:")
    object_d.info()
    object_c.info()
    object_b.info()
    print("Is same object(s)?")
    print(object_d is object_c)
    print(object_d is object_b)
    print(object_c is object_b)
    print("Is untouched original different or same?")
    print(object_d is object_a)
    print("What if we return self?")
    print("Printing Pre-Objects:")
    print(object_b)
    print(object_c)
    print(object_d)
    new_b = object_b.self_method()
    new_c = object_c.self_method()
    new_d = object_d.self_method()
    print(new_b)
    print(new_c)
    print(new_d)
    print("Is same?")
    print(new_d is new_c)
    print(new_d is new_b)
    print(new_c is new_b)
    print("What about the base?")
    og_b1 = object_b.drone
    og_b2 = new_b.drone
    og_d1 = object_d.drone
    og_d2 = new_d.drone
    print(og_b1)
    print(og_b2)
    print(og_d1)
    print(og_d2)
    print("Is same?")
    print(og_b1 is og_b2)
    print(og_d1 is og_d2)
    print(og_b1 is og_d1)
    print(og_b2 is og_d2)
    print(og_b2 is og_d1)
    print(og_b1 is og_d2)
    print("What if we return the protected self?")
    real_b1 = og_b1.queen
    real_b2 = og_b2.queen
    real_b3 = object_b.queen
    real_b4 = new_b.queen
    real_d1 = og_d1.queen
    real_d2 = og_d2.queen
    print(real_b1, real_b2, real_b3, real_b4, real_d1, real_d2)
    print(real_b1 is real_b2)
    print(real_b2 is real_b3)
    print(real_b4 is real_b3)
    print(real_d1 is real_b4)
    print(real_d2 is real_d1)
    print("Test complete!")


if __name__ == "__main__":
    collective_test()
