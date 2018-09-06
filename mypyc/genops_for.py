"""Helpers for generating for loops."""

from mypy.nodes import Lvalue
from mypyc.ops import Value, BasicBlock, is_short_int_rprimitive, LoadInt
from mypyc.ops_int import unsafe_short_add
import mypyc.genops


class For:
    def __init__(self,
                 builder: 'mypyc.genops.IRBuilder',
                 index: Lvalue,
                 body_block: BasicBlock,
                 increment_block: BasicBlock,
                 normal_loop_exit: BasicBlock,
                 line: int) -> None:
        self.builder = builder
        self.index = index
        self.body_block = body_block
        self.increment_block = increment_block
        self.normal_loop_exit = normal_loop_exit
        self.line = line

    def check(self) -> None: ...
    def next(self) -> None: ...


class ForIterable(For):
    def init(self) -> None:
        ...

    def check(self) -> None:
        ...

    def next(self) -> None:
        ...


class ForList(For):
    def init(self) -> None:
        ...

    def check(self) -> None:
        ...

    def next(self) -> None:
        ...


class ForRange(For):
    def init(self, start_reg: Value, end_reg: Value) -> None:
        builder = self.builder
        self.start_reg = start_reg
        self.end_reg = end_reg
        self.end_target = builder.maybe_spill(end_reg)
        # Initialize loop index to 0. Assert that the index target is assignable.
        self.index_target = builder.get_assignment_target(
            self.index)  # type: Union[Register, AssignmentTarget]
        builder.assign(self.index_target, start_reg, self.line)

    def check(self) -> None:
        builder = self.builder
        line = self.line
        # Add loop condition check.
        comparison = builder.binary_op(builder.read(self.index_target, line),
                                       builder.read(self.end_target, line), '<', line)
        builder.add_bool_branch(comparison, self.body_block, self.normal_loop_exit)

    def next(self) -> None:
        builder = self.builder
        line = self.line

        # Increment index register. If the range is known to fit in short ints, use
        # short ints.
        if (is_short_int_rprimitive(self.start_reg.type)
                and is_short_int_rprimitive(self.end_reg.type)):
            new_val = builder.primitive_op(
                unsafe_short_add, [builder.read(self.index_target, line),
                                   builder.add(LoadInt(1))], line)
        else:
            new_val = builder.binary_op(
                builder.read(index_target, line), builder.add(LoadInt(1)), '+', line)
        builder.assign(self.index_target, new_val, line)


class ForCounterInfinite(For):
    def init(self) -> None:
        ...

    def check(self) -> None:
        ...

    def next(self) -> None:
        ...
