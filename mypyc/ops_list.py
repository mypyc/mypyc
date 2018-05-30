import textwrap
from typing import List

from mypyc.ops import (
    int_rprimitive, list_rprimitive, ERR_MAGIC, EmitterInterface, PrimitiveOp2, Register
)
from mypyc.ops_primitive import binary_op


def emit_multiply_helper(emitter: EmitterInterface, dest_reg: Register, list_reg: Register,
                         num_reg: Register) -> None:
    temp = emitter.temp_name()
    num = emitter.reg(num_reg)
    lst = emitter.reg(list_reg)
    dest = emitter.reg(dest_reg)
    emitter.emit_declaration('long long %s;' % temp)
    emitter.emit_lines(
        "%s = CPyTagged_AsLongLong(%s);" % (temp, num),
        "if (%s == -1 && PyErr_Occurred())" % temp,
        "    CPyError_OutOfMemory();",
        "%s = PySequence_Repeat(%s, %s);" % (dest, lst, temp))


def emit_multiply(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    emit_multiply_helper(emitter, op.dest, op.args[0], op.args[1])


def emit_multiply_reversed(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    emit_multiply_helper(emitter, op.dest, op.args[1], op.args[0])


binary_op(op='*',
          arg_types=[list_rprimitive, int_rprimitive],
          result_type=list_rprimitive,
          error_kind=ERR_MAGIC,
          format_str='{dest} = {args[0]} * {args[1]} :: list',
          emit=emit_multiply)

binary_op(op='*',
          arg_types=[int_rprimitive, list_rprimitive],
          result_type=list_rprimitive,
          error_kind=ERR_MAGIC,
          format_str='{dest} = {args[0]} * {args[1]} :: list',
          emit=emit_multiply_reversed)
