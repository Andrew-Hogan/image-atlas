import functools


_ACTIVE_CLASS = "_active_descriptor"
_PROTECTED_SELF = "_self_pointer"


def resistance_is_futile(_wrapped_new=None, *, default_class=None):
    if default_class is None:
        default_class = ShapeDescriptor

    def existing_collective_conformer(wrapped_new):
        @functools.wraps(wrapped_new)
        def new_borg_pod_member(cls, existing_object=None, *, _base_class=default_class):
            if existing_object is None:
                existing_object = _base_class({})
            return wrapped_new(cls, existing_object)
        return new_borg_pod_member

    if _wrapped_new is None:
        return existing_collective_conformer
    return existing_collective_conformer(_wrapped_new)


def assimilate(wrapped_class):
    @resistance_is_futile
    def _borg_pod(cls, existing_object, *args, **kwargs):
        return cls._assimilate_into_object_id(existing_object, *args, **kwargs)

    def _add_distinctiveness(self, shared_state, *args, **kwargs):
        super(self.__class__, self).__init__(shared_state)
        self._active_descriptor = self
        __protected_init__(self, *args, **kwargs)

    wrapped_class.__new__ = _borg_pod

    wrapped_class.__init__, __protected_init__ = _add_distinctiveness, wrapped_class.__init__
    return wrapped_class


class ShapeDescriptor(object):
    def __init__(self, _shared_state=None):
        print("BASE INIT CALLED")
        self.__dict__ = _shared_state if _shared_state is not None else {}
        self._active_descriptor = self
        if _PROTECTED_SELF not in self.__dict__:
            self._protected_self = self

    @property
    def queen(self):
        return self._protected_self

    @classmethod
    def _assimilate_into_object_id(cls, existing_object, *args, **kwargs):
        shared_state = existing_object.__dict__
        hidden_instance = super(cls, cls).__new__(cls)
        hidden_instance.__init__(shared_state, *args, **kwargs)
        return existing_object

    def base_self_method(self):
        return self

    def __getattr__(self, name):
        print(name)
        if _ACTIVE_CLASS in self.__dict__:
            active_class = self.__dict__[_ACTIVE_CLASS]
            if hasattr(active_class, name):
                return getattr(active_class, name)
            if name in self.__dict__:
                return self.__dict__[name]
            raise AttributeError
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError


@assimilate
class Circle(ShapeDescriptor):
    def __init__(self):
        print("CIRCLE INIT CALLED")

    @staticmethod
    def info():
        print("I AM CIRCLE.")

    def self_method(self):
        return self


@assimilate
class AlphaNumeric(ShapeDescriptor):
    def __init__(self):
        print("CHAR INIT CALLED")

    @staticmethod
    def info():
        print("I AM CHARACTER.")

    def self_method(self):
        return self


@assimilate
class Punctuation(ShapeDescriptor):
    def __init__(self):
        print("PUNCT INIT CALLED")

    @staticmethod
    def info():
        print("I AM PUNCTUATION.")

    def self_method(self):
        return self


def collective_test():
    print("Begin test!")
    object_a = ShapeDescriptor()
    object_b = ShapeDescriptor()
    print("Is same object?")
    print(object_b is object_a)
    object_c = Circle(object_b)
    print("New info:")
    object_c.info()
    object_b.info()
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
    og_b1 = object_b.base_self_method()
    og_b2 = new_b.base_self_method()
    og_d1 = object_d.base_self_method()
    og_d2 = new_d.base_self_method()
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
