"""Primitive dict ops."""

from mypyc.ops import (
    EmitterInterface, PrimitiveOp, dict_rprimitive, object_rprimitive, bool_rprimitive, ERR_FALSE,
    ERR_MAGIC
)
from mypyc.ops_primitive import method_op, binary_op, func_op


def emit_get_item(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    dest = emitter.reg(op.dest)
    obj = emitter.reg(op.args[0])
    key = emitter.reg(op.args[1])
    emitter.emit_lines('%s = PyDict_GetItemWithError(%s, %s);' % (dest, obj, key),
                       'if (!%s)' % dest,
                       '    PyErr_SetObject(PyExc_KeyError, %s);' % key,
                       'else',
                       '    Py_INCREF(%s);' % dest)


dict_get_item_op = method_op('builtins.dict.__getitem__',
                             arg_types=[dict_rprimitive, object_rprimitive],
                             result_type=object_rprimitive,
                             error_kind=ERR_MAGIC,
                             emit=emit_get_item)


def emit_set_item(emitter: EmitterInterface, op: PrimitiveOp) -> None:
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


def emit_in(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    temp = emitter.temp_name()
    dest = emitter.reg(op.dest)
    emitter.emit_lines('int %s = PyDict_Contains(%s, %s);' % (temp, emitter.reg(op.args[1]),
                                                              emitter.reg(op.args[0])),
                       'if (%s < 0)' % temp,
                       '    %s = %s;' % (dest, bool_rprimitive.c_error_value()),
                       'else',
                       '    %s = %s;' % (dest, temp))


binary_op(op='in',
          arg_types=[object_rprimitive, dict_rprimitive],
          result_type=bool_rprimitive,
          error_kind=ERR_MAGIC,
          format_str='{dest} = {args[0]} in {args[1]} :: dict',
          emit=emit_in)


def emit_update(emitter: EmitterInterface, op: PrimitiveOp) -> None:
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


def emit_new(emitter: EmitterInterface, op: PrimitiveOp) -> None:
    assert op.dest is not None
    emitter.emit_line('%s = PyDict_New();' % emitter.reg(op.dest))


new_dict_op = func_op(name='builtins.dict',
                      arg_types=[],
                      result_type=dict_rprimitive,
                      error_kind=ERR_MAGIC,
                      emit=emit_new,
                      format_str='{dest} = {{}}')
