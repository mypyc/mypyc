"""Miscellaneous primitive ops."""

from mypyc.ops import EmitterInterface, PrimitiveOp, none_rprimitive, bool_rprimitive, ERR_NEVER
from mypyc.ops_primitive import name_ref_op


def emit_none(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    dest = emitter.reg(op.dest)
    emitter.emit_lines(
        '{} = Py_None;'.format(dest),
        'Py_INCREF({});'.format(dest),
    )


none_op = name_ref_op('builtins.None',
                      result_type=none_rprimitive,
                      error_kind=ERR_NEVER,
                      emit=emit_none)


def emit_true(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    emitter.emit_line('%s = 1;' % emitter.reg(op.dest))


true_op = name_ref_op('builtins.True',
                      result_type=bool_rprimitive,
                      error_kind=ERR_NEVER,
                      emit=emit_true)


def emit_false(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    emitter.emit_line('%s = 0;' % emitter.reg(op.dest))


false_op = name_ref_op('builtins.False',
                       result_type=bool_rprimitive,
                       error_kind=ERR_NEVER,
                       emit=emit_false)
