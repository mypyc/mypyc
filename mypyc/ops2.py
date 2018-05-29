from typing import Dict, List, Callable

from mypyc.ops import (
    OpDescription, PrimitiveOp2, int_rprimitive, RType, EmitterInterface, ERR_NEVER
)


binary_ops = {}  # type: Dict[str, List[OpDescription]]


def binary_op(op: str,
              arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              format_str: str,
              emit: Callable[[EmitterInterface, PrimitiveOp2], None]) -> None:
    ops = binary_ops.setdefault(op, [])
    desc = OpDescription(op, arg_types, result_type, error_kind, format_str, emit)
    ops.append(desc)


def int_binary_op(op: str, c_func_name: str) -> None:
    def emit(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
        assert op.dest is not None
        line = '%s = %s(%s, %s);' % (emitter.reg(op.dest), c_func_name,
                                     emitter.reg(op.args[0]), emitter.reg(op.args[1]))
        emitter.emit_line(line)

    binary_op(op=op,
              arg_types=[int_rprimitive, int_rprimitive],
              result_type=int_rprimitive,
              error_kind=ERR_NEVER,
              format_str='{dest} = {args[0]} %s {args[1]} :: int' % op,
              emit=emit)


int_binary_op('+', 'CPyTagged_Add')
int_binary_op('-', 'CPyTagged_Subtract')
int_binary_op('*', 'CPyTagged_Multiply')
int_binary_op('//', 'CPyTagged_FloorDivide')
int_binary_op('%', 'CPyTagged_Remainder')
