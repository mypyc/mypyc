from typing import Optional, List

from mypyc.ops import FuncIR, BasicBlock, Label, LoadErrorValue, Return, Goto, Branch


def insert_exception_handling(ir: FuncIR) -> None:
    for block in ir.blocks:
        insert_exception_handling_to_block(block, ir)
    ir.blocks = split_blocks_at_errors(ir.blocks)


def insert_exception_handling_to_block(block: BasicBlock, ir: FuncIR) -> None:
    handler_block = None  # type: Optional[Label]
    for op in block.ops:
        if op.can_raise():
            if not handler_block:
                handler_block = add_handler_block(ir)
            op.error_label = handler_block


def add_handler_block(ir: FuncIR) -> Label:
    block = BasicBlock(Label(len(ir.blocks)))
    rtype = ir.ret_type
    reg = ir.env.add_temp(rtype)
    block.ops.append(LoadErrorValue(reg, rtype))
    block.ops.append(Return(reg))
    ir.blocks.append(block)
    return block.label


def split_blocks_at_errors(blocks: List[BasicBlock]) -> List[BasicBlock]:
    new_blocks = []
    mapping = {}
    # First split blocks.
    for block in blocks:
        ops = block.ops
        i0 = 0
        i = 0
        new_ops = []
        while i < len(ops) - 1:
            if ops[i].error_label is not None:
                # Split
                new_block = BasicBlock(len(new_blocks))
                new_block.ops.extend(ops[i0:i + 1])
                new_block.ops[-1].ok_label = new_block.label + 1
                mapping[block.label] = new_block.label
                new_blocks.append(new_block)
                i += 1
                i0 = i
            else:
                i += 1
        new_block = BasicBlock(len(new_blocks))
        new_block.ops.extend(ops[i0:i + 1])
        mapping[block.label] = new_block.label
        new_blocks.append(new_block)
    # Adjust all labels to reflect the new blocks.
    for block in new_blocks:
        for op in block.ops:
            if op.error_label is not None:
                op.error_label = mapping[op.error_label]
            elif isinstance(op, Goto):
                op.label = mapping[op.label]
            elif isinstance(op, Branch):
                op.true = mapping[op.true]
                op.false = mapping[op.false]
    return new_blocks
