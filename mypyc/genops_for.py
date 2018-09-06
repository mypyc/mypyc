"""Helpers for generating for loops."""

from typing import Union, List

from mypy.nodes import Lvalue, Expression
from mypyc.ops import (
    Value, BasicBlock, is_short_int_rprimitive, LoadInt, RType, PrimitiveOp, Branch, Register,
    AssignmentTarget
)
from mypyc.ops_int import unsafe_short_add
from mypyc.ops_list import list_len_op
from mypyc.ops_misc import iter_op, next_op
from mypyc.ops_exc import no_err_occurred_op
import mypyc.genops


class ForGenerator:
    def __init__(self,
                 builder: 'mypyc.genops.IRBuilder',
                 index: Lvalue,
                 body_block: BasicBlock,
                 loop_exit: BasicBlock,
                 line: int) -> None:
        self.builder = builder
        self.index = index
        self.body_block = body_block
        self.loop_exit = loop_exit
        self.line = line

    def add_cleanup(self, exit_block: BasicBlock) -> None:
        if self.need_cleanup():
            self.builder.activate_block(self.loop_exit)
            self.cleanup()
            self.builder.goto(exit_block)

    def has_combined_next_and_check(self) -> bool:
        return False

    def need_cleanup(self) -> bool:
        return False

    def check(self) -> None:
        pass

    def begin_body(self) -> None:
        pass

    def next(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


class ForIterable(ForGenerator):
    """Generate IR for a for loop over an arbitrary iterable."""

    def has_combined_next_and_check(self) -> bool:
        # We always need to get the next item before doing a check.
        return True

    def need_cleanup(self) -> bool:
        # Create a new cleanup block for when the loop is finished.
        return True

    def init(self, expr_reg: Value) -> None:
        # Define targets to contain the expression, along with the iterator that will be used
        # for the for-loop. If we are inside of a generator function, spill these into the
        # environment class.
        builder = self.builder
        iter_reg = builder.primitive_op(iter_op, [expr_reg], self.line)
        builder.maybe_spill(expr_reg)
        self.iter_target = builder.maybe_spill(iter_reg)

    def check(self) -> None:
        # We create a block for where the __next__ function will be called on the iterator and
        # checked to see if the value returned is NULL, which would signal either the end of
        # the Iterable being traversed or an exception being raised. Note that Branch.IS_ERROR
        # checks only for NULL (an exception does not necessarily have to be raised).
        builder = self.builder
        line = self.line
        self.next_reg = builder.primitive_op(next_op, [builder.read(self.iter_target, line)], line)
        builder.add(Branch(self.next_reg, self.loop_exit, self.body_block, Branch.IS_ERROR))

    def begin_body(self) -> None:
        # Assign the value obtained from __next__ to the
        # lvalue so that it can be referenced by code in the body of the loop. At the end of
        # the body, we goto the label that calls the iterator's __next__ function again.
        builder = self.builder
        builder.assign(builder.get_assignment_target(self.index), self.next_reg, self.line)

    def next(self) -> None:
        # Nothing to do here, since we get the next item as part of check().
        pass

    def cleanup(self) -> None:
        # We set the branch to go here if the conditional evaluates to true. If
        # an exception was raised during the loop, then err_reg wil be set to
        # True. If no_err_occurred_op returns False, then the exception will be
        # propagated using the ERR_FALSE flag.
        self.builder.primitive_op(no_err_occurred_op, [], self.line)


class ForList(ForGenerator):
    """Generate IR for a for loop over a list."""

    def init(self, expr_reg: Value, target_type: RType) -> None:
        builder = self.builder
        # Define target to contain the expression, along with the index that will be used
        # for the for-loop. If we are inside of a generator function, spill these into the
        # environment class.
        index_reg = builder.add(LoadInt(0))
        self.expr_target = builder.maybe_spill(expr_reg)
        self.index_target = builder.maybe_spill_assignable(index_reg)
        self.target_type = target_type

    def check(self) -> None:
        builder = self.builder
        line = self.line
        # For compatibility with python semantics we recalculate the length
        # at every iteration.
        len_reg = builder.add(PrimitiveOp([builder.read(self.expr_target, line)],
                                          list_len_op, line))
        comparison = builder.binary_op(builder.read(self.index_target, line), len_reg, '<', line)
        builder.add_bool_branch(comparison, self.body_block, self.loop_exit)

    def begin_body(self) -> None:
        builder = self.builder
        line = self.line
        value_box = builder.translate_special_method_call(
            builder.read(self.expr_target, line), '__getitem__',
            [builder.read(self.index_target, line)], None, line)
        assert value_box

        builder.assign(builder.get_assignment_target(self.index),
                       builder.unbox_or_cast(value_box, self.target_type, line), line)

    def next(self) -> None:
        builder = self.builder
        line = self.line
        builder.assign(self.index_target, builder.primitive_op(
            unsafe_short_add,
            [builder.read(self.index_target, line),
             builder.add(LoadInt(1))], line), line)


class ForRange(ForGenerator):
    """Generate IR for a for loop over an integer range."""

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
        builder.add_bool_branch(comparison, self.body_block, self.loop_exit)

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
                builder.read(self.index_target, line), builder.add(LoadInt(1)), '+', line)
        builder.assign(self.index_target, new_val, line)


class ForInfiniteCounter(ForGenerator):
    """Generate IR for a for loop counting from 0 to infinity."""

    def init(self) -> None:
        builder = self.builder
        # Initialize loop index to 0. Assert that the index target is assignable.
        self.index_target = builder.get_assignment_target(
            self.index)  # type: Union[Register, AssignmentTarget]
        builder.assign(self.index_target, builder.add(LoadInt(0)), self.line)

    def next(self) -> None:
        builder = self.builder
        line = self.line
        new_val = builder.primitive_op(
            unsafe_short_add, [builder.read(self.index_target, line),
                               builder.add(LoadInt(1))], line)
        builder.assign(self.index_target, new_val, line)


class ForEnumerate(ForGenerator):
    """Generate IR for a for loop of form `for i, x in enumerate(...)`."""

    def need_cleanup(self) -> bool:
        # The wrapped for loop might need cleanup. We might generate a
        # redundant cleanup block, but that's okay.
        return True

    def init(self, index1: Lvalue, index2: Lvalue, expr: Expression) -> None:
        # Count from 0 to infinity.
        self.index_gen = ForInfiniteCounter(
            self.builder,
            index1,
            self.body_block,
            self.loop_exit,
            self.line)
        self.index_gen.init()
        # Iterate over the actual iterable.
        self.main_gen = self.builder.make_for_loop_generator(
            index2,
            expr,
            self.body_block,
            self.loop_exit,
            self.line)

    def check(self) -> None:
        # No need for a check for the index generator, since it's unconditional.
        self.main_gen.check()

    def begin_body(self) -> None:
        self.index_gen.begin_body()
        self.main_gen.begin_body()

    def next(self) -> None:
        self.index_gen.next()
        self.main_gen.next()

    def cleanup(self) -> None:
        self.index_gen.cleanup()
        self.main_gen.cleanup()


class ForZip(ForGenerator):
    """Generate IR for a for loop of form `for x, y in zip(a, b)`."""

    def need_cleanup(self) -> bool:
        # The wrapped for loop might need cleanup. We might generate a
        # redundant cleanup block, but that's okay.
        return True

    def init(self, indexes: List[Lvalue], exprs: List[Expression]) -> None:
        assert len(indexes) == len(exprs)
        self.cond_blocks = [BasicBlock() for _ in range(len(indexes) - 1)]
        self.cond_blocks.append(self.body_block)
        self.gens = []  # type: List[ForGenerator]
        for index, expr, next_block in zip(indexes, exprs, self.cond_blocks):
            gen = self.builder.make_for_loop_generator(
                index,
                expr,
                next_block,
                self.loop_exit,
                self.line, reuse_cleanup=True)
            self.gens.append(gen)

    def check(self) -> None:
        # No need for a check for the index generator, since it's unconditional.
        for i, gen in enumerate(self.gens):
            gen.check()
            if i < len(self.gens) - 1:
                self.builder.activate_block(self.cond_blocks[i])

    def begin_body(self) -> None:
        for gen in self.gens:
            gen.begin_body()

    def next(self) -> None:
        for gen in self.gens:
            gen.next()

    def cleanup(self) -> None:
        for gen in self.gens:
            gen.cleanup()
