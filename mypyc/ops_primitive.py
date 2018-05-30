"""Utilities for defining primitive ops."""

from typing import Dict, List, Callable, Optional

from mypyc.ops import (
    OpDescription, PrimitiveOp2, RType, EmitterInterface, short_name
)


binary_ops = {}  # type: Dict[str, List[OpDescription]]
unary_ops = {}  # type: Dict[str, List[OpDescription]]
func_ops = {}  # type: Dict[str, List[OpDescription]]
method_ops = {}  # type: Dict[str, List[OpDescription]]

EmitCallback = Callable[[EmitterInterface, PrimitiveOp2], None]


def binary_op(op: str,
              arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              format_str: str,
              emit: EmitCallback) -> None:
    assert len(arg_types) == 2
    ops = binary_ops.setdefault(op, [])
    desc = OpDescription(op, arg_types, result_type, error_kind, format_str, emit)
    ops.append(desc)


def unary_op(op: str,
             arg_type: RType,
             result_type: RType,
             error_kind: int,
             format_str: str,
             emit: EmitCallback) -> None:
    ops = unary_ops.setdefault(op, [])
    desc = OpDescription(op, [arg_type], result_type, error_kind, format_str, emit)
    ops.append(desc)


def func_op(name: str,
            arg_types: List[RType],
            result_type: RType,
            error_kind: int,
            emit: EmitCallback) -> OpDescription:
    ops = func_ops.setdefault(name, [])
    typename = ''
    if len(arg_types) == 1:
        typename = ' :: %s' % short_name(arg_types[0].name)
    format_str = '{dest} = %s %s%s' % (short_name(name),
                                       ', '.join('{args[%d]}' % i
                                                 for i in range(len(arg_types))),
                                       typename)
    desc = OpDescription(name, arg_types, result_type, error_kind, format_str, emit)
    ops.append(desc)
    return desc


def method_op(name: str,
              arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              emit: EmitCallback) -> OpDescription:
    ops = method_ops.setdefault(name, [])
    assert len(arg_types) > 0
    args = ', '.join('{args[%d]}' % i
                     for i in range(1, len(arg_types)))
    method_name = name.rpartition('.')[2]
    format_str = '{dest} = {args[0]}.%s(%s)' % (method_name, args)
    desc = OpDescription(method_name, arg_types, result_type, error_kind, format_str, emit)
    ops.append(desc)
    return desc
