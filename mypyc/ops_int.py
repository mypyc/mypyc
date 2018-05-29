from mypyc.ops import PrimitiveOp2, int_rprimitive, RType, EmitterInterface, ERR_NEVER
from mypyc.ops_primitive import binary_op, unary_op


def int_binary_op(op: str, c_func_name: str) -> None:
    def emit(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
        assert op.dest is not None
        line = '%s = %s(%s, %s);' % (emitter.reg(op.dest), c_func_name,
                                     emitter.reg(op.args[0]), emitter.reg(op.args[1]))
        emitter.emit_line(line)

    binary_op(op=op,
              arg_types=[int_rprimitive, int_rprimitive],
              result_type=int_rprimitive,
              error_kind=ERR_NEVER,
              format_str='{dest} = {args[0]} %s {args[1]} :: int' % op,
              emit=emit)


int_binary_op('+', 'CPyTagged_Add')
int_binary_op('-', 'CPyTagged_Subtract')
int_binary_op('*', 'CPyTagged_Multiply')
int_binary_op('//', 'CPyTagged_FloorDivide')
int_binary_op('%', 'CPyTagged_Remainder')


def int_unary_op(op: str, c_func_name: str) -> None:
    def emit(emitter: EmitterInterface, op: PrimitiveOp2) -> None:
        assert op.dest is not None
        line = '%s = %s(%s);' % (emitter.reg(op.dest), c_func_name, emitter.reg(op.args[0]))
        emitter.emit_line(line)

    unary_op(op=op,
             arg_type=int_rprimitive,
             result_type=int_rprimitive,
             error_kind=ERR_NEVER,
             format_str='{dest} = %s{args[0]} :: int' % op,
             emit=emit)


int_unary_op('-', 'CPyTagged_Negate')
