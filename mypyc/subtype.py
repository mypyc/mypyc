"""Subtype check for RTypes."""

from mypyc.ops import (
    RType, OptionalRType, UserRType, RInstance, TupleRType, RTypeVisitor,
    is_bool_rinstance, is_int_rinstance, is_tuple_rinstance, none_rinstance, is_object_rinstance
)


def is_subtype(left: RType, right: RType) -> bool:
    if is_object_rinstance(right):
        return True
    elif isinstance(right, OptionalRType):
        if is_subtype(left, none_rinstance) or is_subtype(left, right.value_type):
            return True
    return left.accept(SubtypeVisitor(right))


class SubtypeVisitor(RTypeVisitor[bool]):
    """Is left a subtype of right?

    A few special cases such as right being 'object' are handled in
    is_subtype and don't need to be covered here.
    """

    def __init__(self, right: RType) -> None:
        self.right = right

    def visit_user_rtype(self, left: UserRType) -> bool:
        # TODO: Inheritance
        return isinstance(self.right, UserRType) and self.right.name == left.name

    def visit_optional_rtype(self, left: OptionalRType) -> bool:
        return isinstance(self.right, OptionalRType) and is_subtype(left.value_type,
                                                                    self.right.value_type)

    def visit_rinstance(self, left: RInstance) -> bool:
        if is_bool_rinstance(left) and is_int_rinstance(self.right):
            return True
        return isinstance(self.right, RInstance) and left.name == self.right.name

    def visit_tuple_rtype(self, left: TupleRType) -> bool:
        if is_tuple_rinstance(self.right):
            return True
        if isinstance(self.right, TupleRType):
            return len(self.right.types) == len(left.types) and all(
                is_subtype(t1, t2) for t1, t2 in zip(left.types, self.right.types))
        return False
