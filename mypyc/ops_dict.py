"""Primitive dict ops."""

from mypyc.ops import (
    EmitterInterface, PrimitiveOp2, dict_rprimitive, object_rprimitive, bool_rprimitive, ERR_FALSE
)
from mypyc.ops_primitive import method_op


def emit_set_item(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    emitter.emit_line('%s = PyDict_SetItem(%s, %s, %s) >= 0;' % (emitter.reg(op.dest),
                                                                 emitter.reg(op.args[0]),
                                                                 emitter.reg(op.args[1]),
                                                                 emitter.reg(op.args[2])))


dict_set_item_op = method_op(name='builtins.dict.__setitem__',
                             arg_types=[dict_rprimitive, object_rprimitive, object_rprimitive],
                             result_type=bool_rprimitive,
                             error_kind=ERR_FALSE,
                             emit=emit_set_item)


def emit_update(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
    assert op.dest is not None
    # NOTE: PyDict_Update is technically not equivalent to update, but the cases where it
    # differs (when the second argument has no keys) should never typecheck for us, so the
    # difference is irrelevant.
    emitter.emit_line(
        '%s = PyDict_Update(%s, %s) != -1;' % (emitter.reg(op.dest),
                                               emitter.reg(op.args[0]),
                                               emitter.reg(op.args[1])))


dict_update_op = method_op(name='builtins.dict.update',
                           arg_types=[dict_rprimitive, object_rprimitive],
                           result_type=None,
                           error_kind=ERR_FALSE,
                           emit=emit_update)
