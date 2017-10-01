from typing import Optional

from mypyc.ops import FuncIR, BasicBlock, Label, LoadErrorValue, Return


def insert_exception_handling(ir: FuncIR) -> None:
    for block in ir.blocks:
        insert_exception_handling_to_block(block, ir)


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
