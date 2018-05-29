"""Utilities for defining primitive ops."""

from typing import Dict, List, Callable

from mypyc.ops import (
    OpDescription, PrimitiveOp2, RType, EmitterInterface
)


binary_ops = {}  # type: Dict[str, List[OpDescription]]


def binary_op(op: str,
              arg_types: List[RType],
              result_type: RType,
              error_kind: int,
              format_str: str,
              emit: Callable[[EmitterInterface, PrimitiveOp2], None]) -> None:
    ops = binary_ops.setdefault(op, [])
    desc = OpDescription(op, arg_types, result_type, error_kind, format_str, emit)
    ops.append(desc)
