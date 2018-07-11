"""Exception-related primitive ops."""

from typing import List

from mypyc.ops import (
    EmitterInterface, PrimitiveOp, none_rprimitive, bool_rprimitive, object_rprimitive,
    void_rtype,
    exc_rtuple,
    ERR_NEVER, ERR_MAGIC, ERR_FALSE
)
from mypyc.ops_primitive import (
    simple_emit, func_op, method_op, custom_op,
    negative_int_emit,
)

# TODO: Making this raise conditionally is kind of hokey.
raise_exception_op = custom_op(
    arg_types=[object_rprimitive, object_rprimitive],
    result_type=bool_rprimitive,
    error_kind=ERR_FALSE,
    format_str = 'raise_exception({args[0]}, {args[1]}); {dest} = 0',
    emit=simple_emit('PyErr_SetObject({args[0]}, {args[1]}); {dest} = 0;'))

reraise_exception_op = custom_op(
    arg_types=[],
    result_type=bool_rprimitive,
    error_kind=ERR_FALSE,
    format_str = 'reraise_exc; {dest} = 0',
    emit=simple_emit('CPy_Reraise(); {dest} = 0;'))

clear_exception_op = custom_op(
    arg_types=[],
    result_type=void_rtype,
    error_kind=ERR_NEVER,
    format_str = 'clear_exception',
    emit=simple_emit('PyErr_Clear();'))

no_err_occurred_op = custom_op(
    arg_types=[],
    result_type=bool_rprimitive,
    error_kind=ERR_FALSE,
    format_str = '{dest} = no_err_occurred',
    emit=simple_emit('{dest} = (PyErr_Occurred() == NULL);'))

error_catch_op = custom_op(
    arg_types=[],
    result_type=exc_rtuple,
    error_kind=ERR_NEVER,
    format_str = '{dest} = err_catch',
    emit=simple_emit('CPy_CatchError(&{dest}.f0, &{dest}.f1, &{dest}.f2);'))

restore_exc_info_op = custom_op(
    arg_types=[exc_rtuple],
    result_type=void_rtype,
    error_kind=ERR_NEVER,
    format_str = 'restore_exc_info {args[0]}',
    emit=simple_emit('CPy_RestoreExcInfo({args[0]}.f0, {args[0]}.f1, {args[0]}.f2);'))

exc_matches_op = custom_op(
    arg_types=[object_rprimitive],
    result_type=bool_rprimitive,
    error_kind=ERR_NEVER,
    format_str = '{dest} = exc_matches {args[0]}',
    emit=simple_emit('{dest} = CPy_ExceptionMatches({args[0]});'))

get_exc_value_op = custom_op(
    arg_types=[],
    result_type=object_rprimitive,
    error_kind=ERR_NEVER,
    format_str = '{dest} = get_exc_value',
    emit=simple_emit('{dest} = CPy_GetExcValue();'))
