"""Utilities for defining primitive ops."""

from typing import Dict, List, Callable, Optional

from mypyc.ops import (
    OpDescription, PrimitiveOp2, RType, EmitterInterface, short_name
)


# Primitive binary ops (key is operator such as '+')
binary_ops = {}  # type: Dict[str, List[OpDescription]]
# Primitive unary ops (key is operator such as '-')
unary_ops = {}  # type: Dict[str, List[OpDescription]]
# Primitive ops for built-in functions (key is function name such as 'builtins.len')
func_ops = {}  # type: Dict[str, List[OpDescription]]
# Primitive ops for built-in methods (key is method name such as 'builtins.list.append')
method_ops = {}  # type: Dict[str, List[OpDescription]]
# Primitive ops for reading module attributes (key is name such as 'builtins.None')
name_ref_ops = {}  # type: Dict[str, OpDescription]

EmitCallback = Callable[[EmitterInterface, PrimitiveOp2], None]


def binary_op(op: str,
              arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              format_str: str,
              emit: EmitCallback) -> None:
    assert len(arg_types) == 2
    ops = binary_ops.setdefault(op, [])
    desc = OpDescription(op, arg_types, result_type, False, error_kind, format_str, emit)
    ops.append(desc)


def unary_op(op: str,
             arg_type: RType,
             result_type: RType,
             error_kind: int,
             format_str: str,
             emit: EmitCallback) -> None:
    ops = unary_ops.setdefault(op, [])
    desc = OpDescription(op, [arg_type], result_type, False, error_kind, format_str, emit)
    ops.append(desc)


def func_op(name: str,
            arg_types: List[RType],
            result_type: RType,
            error_kind: int,
            emit: EmitCallback,
            format_str: Optional[str] = None) -> OpDescription:
    ops = func_ops.setdefault(name, [])
    typename = ''
    if len(arg_types) == 1:
        typename = ' :: %s' % short_name(arg_types[0].name)
    if format_str is None:
        format_str = '{dest} = %s %s%s' % (short_name(name),
                                           ', '.join('{args[%d]}' % i
                                                     for i in range(len(arg_types))),
                                           typename)
    desc = OpDescription(name, arg_types, result_type, False, error_kind, format_str, emit)
    ops.append(desc)
    return desc


def method_op(name: str,
              arg_types: List[RType],
              result_type: Optional[RType],
              error_kind: int,
              emit: EmitCallback) -> OpDescription:
    ops = method_ops.setdefault(name, [])
    assert len(arg_types) > 0
    args = ', '.join('{args[%d]}' % i
                     for i in range(1, len(arg_types)))
    method_name = name.rpartition('.')[2]
    if method_name == '__getitem__':
        format_str = '{dest} = {args[0]}[{args[1]}] :: %s' % short_name(arg_types[0].name)
    else:
        format_str = '{dest} = {args[0]}.%s(%s)' % (method_name, args)
    desc = OpDescription(method_name, arg_types, result_type, False, error_kind, format_str, emit)
    ops.append(desc)
    return desc


def name_ref_op(name: str,
                result_type: RType,
                error_kind: int,
                emit: EmitCallback) -> OpDescription:
    """Define an op that is used to implement reading a module attribute.

    Args:
        name: fully-qualified name (e.g. 'builtins.None')
    """
    assert name not in name_ref_ops, 'already defined: %s' % name
    format_str = '{dest} = %s' % short_name(name)
    desc = OpDescription(name, [], result_type, False, error_kind, format_str, emit)
    name_ref_ops[name] = desc
    return desc


def custom_op(arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              format_str: str,
              emit: EmitCallback,
              is_var_arg: bool = False) -> OpDescription:
    return OpDescription('<custom>', arg_types, result_type, is_var_arg, error_kind, format_str,
                         emit)
