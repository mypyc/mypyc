"""*Runtime* Subtype check for RTypes."""

from mypyc.ops import (
    RType, RUnion, RInstance, RPrimitive, RTuple, RVoid, RTypeVisitor,
    is_bool_rprimitive, is_int_rprimitive, is_tuple_rprimitive, none_rprimitive,
    is_short_int_rprimitive,
    is_object_rprimitive
)
from mypyc.sametype import is_same_type
from mypyc.subtype import is_subtype


def is_runtime_subtype(left: RType, right: RType) -> bool:
    if not left.is_unboxed and is_object_rprimitive(right):
        return True
    return left.accept(RTSubtypeVisitor(right))


class RTSubtypeVisitor(RTypeVisitor[bool]):
    """Is left a runtime subtype of right?

    A few special cases such as right being 'object' are handled in
    is_runtime_subtype and don't need to be covered here.
    """

    def __init__(self, right: RType) -> None:
        self.right = right

    def visit_rinstance(self, left: RInstance) -> bool:
        return isinstance(self.right, RInstance) and is_subtype(left, self.right)

    def visit_runion(self, left: RUnion) -> bool:
        return is_subtype(left, self.right)

    def visit_rprimitive(self, left: RPrimitive) -> bool:
        if is_short_int_rprimitive(left) and is_int_rprimitive(self.right):
            return True
        return left is self.right

    def visit_rtuple(self, left: RTuple) -> bool:
        # We might want to implement runtime subtyping for tuples. The
        # obstacle is that we generate different (but equivalent)
        # tuple structs.
        return is_same_type(left, self.right)

    def visit_rvoid(self, left: RVoid) -> bool:
        return isinstance(self.right, RVoid)
