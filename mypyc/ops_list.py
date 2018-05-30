import textwrap
from typing import List

from mypyc.ops import (
    int_rprimitive, list_rprimitive, object_rprimitive, bool_rprimitive, ERR_MAGIC, ERR_NEVER,
    ERR_FALSE, EmitterInterface, PrimitiveOp2, Register
)
from mypyc.ops_primitive import binary_op, func_op, method_op


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


def emit_len(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    temp = emitter.temp_name()
    emitter.emit_declaration('long long %s;' % temp)
    emitter.emit_line('%s = PyList_GET_SIZE(%s);' % (temp, emitter.reg(op.args[0])))
    emitter.emit_line('%s = CPyTagged_ShortFromLongLong(%s);' % (emitter.reg(op.dest), temp))


list_len_op = func_op(name='builtins.len',
                      arg_types=[list_rprimitive],
                      result_type=int_rprimitive,
                      error_kind=ERR_NEVER,
                      emit=emit_len)


def emit_append(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    emitter.emit_line(
        '%s = PyList_Append(%s, %s) != -1;' % (emitter.reg(op.dest),
                                               emitter.reg(op.args[0]),
                                               emitter.reg(op.args[1])))


list_append_op = method_op(name='builtins.list.append',
                           arg_types=[list_rprimitive, object_rprimitive],
                           result_type=bool_rprimitive,
                           error_kind=ERR_FALSE,
                           emit=emit_append)
