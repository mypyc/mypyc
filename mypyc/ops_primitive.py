"""Utilities for defining primitive ops."""

from typing import Dict, List, Callable

from mypyc.ops import (
    OpDescription, PrimitiveOp2, RType, EmitterInterface
)


binary_ops = {}  # type: Dict[str, List[OpDescription]]
unary_ops = {}  # type: Dict[str, List[OpDescription]]

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
