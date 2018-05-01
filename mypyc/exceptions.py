from typing import Optional, List

from mypyc.ops import (
    FuncIR, BasicBlock, Label, LoadErrorValue, Return, Goto, Branch, ERR_NEVER, ERR_MAGIC,
    ERR_FALSE, INVALID_REGISTER, RegisterOp
)


def insert_exception_handling(ir: FuncIR) -> None:
    error_label = None
    for block in ir.blocks:
        can_raise = any(op.can_raise() for op in block.ops)
        if can_raise:
            error_label = add_handler_block(ir)
            break
    ir.blocks = split_blocks_at_errors(ir.blocks, error_label, ir.name)


def can_raise(block: BasicBlock) -> bool:
    return any(op.can_raise() for op in block.ops)


def add_handler_block(ir: FuncIR) -> Label:
    block = BasicBlock(Label(len(ir.blocks)))
    rtype = ir.ret_type
    reg = ir.env.add_temp(rtype)
    block.ops.append(LoadErrorValue(reg, rtype))
    block.ops.append(Return(reg))
    ir.blocks.append(block)
    return block.label


def split_blocks_at_errors(blocks: List[BasicBlock],
                           error_label: Label,
                           func: str) -> List[BasicBlock]:
    new_blocks = []  # type: List[BasicBlock]
    mapping = {}
    partial_ops = set()
    # First split blocks on ops that may raise.
    for block in blocks:
        ops = block.ops
        i0 = 0
        i = 0
        while i < len(ops) - 1:
            op = ops[i]
            if isinstance(op, RegisterOp) and op.error_kind != ERR_NEVER:
                # Split
                new_block = BasicBlock(Label(len(new_blocks)))
                new_block.ops.extend(ops[i0:i + 1])

                if op.error_kind == ERR_MAGIC:
                    variant = Branch.IS_ERROR
                    negated = False
                elif op.error_kind == ERR_FALSE:
                    variant = Branch.BOOL_EXPR
                    negated = True
                else:
                    assert False, 'unknown error kind %d' % op.error_kind

                # Void ops can't generate errors.
                assert op.dest is not None, op

                branch = Branch(op.dest, INVALID_REGISTER,
                                true_label=error_label,
                                false_label=Label(new_block.label + 1),
                                op=variant)
                branch.negated = negated
                branch.traceback_entry = (func, op.line)
                partial_ops.add(branch)  # Only tweak true label of these
                new_block.ops.append(branch)
                if i0 == 0:
                    mapping[block.label] = new_block.label
                new_blocks.append(new_block)
                i += 1
                i0 = i
            else:
                i += 1
        new_block = BasicBlock(Label(len(new_blocks)))
        new_block.ops.extend(ops[i0:i + 1])
        if i0 == 0:
            mapping[block.label] = new_block.label
        new_blocks.append(new_block)
    # Adjust all labels to reflect the new blocks.
    for block in new_blocks:
        for op in block.ops:
            if isinstance(op, Goto):
                op.label = mapping[op.label]
            elif isinstance(op, Branch):
                if op not in partial_ops:
                    op.false = mapping[op.false]
                op.true = mapping[op.true]
    return new_blocks
