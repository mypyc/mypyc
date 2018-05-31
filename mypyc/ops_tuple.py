"""Primitive (varying-length) tuple ops."""

from mypyc.ops import EmitterInterface, PrimitiveOp2, tuple_rprimitive, int_rprimitive, ERR_NEVER
from mypyc.ops_primitive import func_op


def emit_len(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    temp = emitter.temp_name()
    emitter.emit_declaration('long long %s;' % temp)
    emitter.emit_line('%s = PyTuple_GET_SIZE(%s);' % (temp, emitter.reg(op.args[0])))
    emitter.emit_line('%s = CPyTagged_ShortFromLongLong(%s);' % (emitter.reg(op.dest), temp))


tuple_len_op = func_op(name='builtins.len',
                       arg_types=[tuple_rprimitive],
                       result_type=int_rprimitive,
                       error_kind=ERR_NEVER,
                       emit=emit_len)
