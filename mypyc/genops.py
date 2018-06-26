"""Transform a mypy AST to the IR form (Intermediate Representation).

For example, consider a function like this:

   def f(x: int) -> int:
       return x * 2 + 1

It would be translated to something that conceptually looks like this:

   r0 = 2
   r1 = 1
   r2 = x * i0 :: int
   r3 = r2 + r1 :: int
   return r3
"""

from typing import Dict, List, Tuple, Optional, Union, Sequence

from mypy.nodes import (
    Node, MypyFile, SymbolNode, FuncDef, ReturnStmt, AssignmentStmt, OpExpr, IntExpr, NameExpr,
    LDEF, Var, IfStmt, UnaryExpr, ComparisonExpr, WhileStmt, Argument, CallExpr, IndexExpr, Block,
    Expression, ListExpr, ExpressionStmt, MemberExpr, ForStmt, RefExpr, Lvalue, BreakStmt,
    ContinueStmt, ConditionalExpr, OperatorAssignmentStmt, TupleExpr, ClassDef, TypeInfo,
    Import, ImportFrom, ImportAll, DictExpr, StrExpr, CastExpr, TempNode, ARG_POS, MODULE_REF,
    PassStmt, PromoteExpr, AwaitExpr, BackquoteExpr, AssertStmt, BytesExpr, ComplexExpr,
    Decorator, DelStmt, DictionaryComprehension, EllipsisExpr, EnumCallExpr, ExecStmt,
    FloatExpr, GeneratorExpr, GlobalDecl, LambdaExpr, ListComprehension, SetComprehension,
    NamedTupleExpr, NewTypeExpr, NonlocalDecl, OverloadedFuncDef, PrintStmt, RaiseStmt,
    RevealExpr, SetExpr, SliceExpr, StarExpr, SuperExpr, TryStmt, TypeAliasExpr,
    TypeApplication, TypeVarExpr, TypedDictExpr, UnicodeExpr, WithStmt, YieldFromExpr, YieldExpr,
    GDEF
)
import mypy.nodes
from mypy.types import (
    Type, Instance, CallableType, NoneTyp, TupleType, UnionType, AnyType, TypeVarType,
)
from mypy.visitor import NodeVisitor
from mypy.subtypes import is_named_instance
from mypy.checkmember import bind_self

from mypyc.common import MAX_SHORT_INT
from mypyc.ops import (
    BasicBlock, AssignmentTarget, AssignmentTargetRegister, AssignmentTargetIndex,
    AssignmentTargetAttr, Environment, Op, LoadInt, RType, Value, Register, Label, Return, FuncIR,
    Assign, Branch, Goto, RuntimeArg, Call, Box, Unbox, Cast, RTuple, Unreachable, TupleGet,
    TupleSet, ClassIR, RInstance, ModuleIR, GetAttr, SetAttr, LoadStatic, PyCall, ROptional,
    PyMethodCall, MethodCall, INVALID_VALUE, INVALID_LABEL, int_rprimitive,
    is_int_rprimitive, float_rprimitive, is_float_rprimitive, bool_rprimitive, list_rprimitive,
    is_list_rprimitive, dict_rprimitive, is_dict_rprimitive, str_rprimitive, is_tuple_rprimitive,
    tuple_rprimitive, none_rprimitive, is_none_rprimitive, object_rprimitive, PrimitiveOp,
    ERR_FALSE, OpDescription, RegisterOp, is_object_rprimitive, LiteralsMap, FuncSignature,
    VTableAttr, VTableMethod, NAMESPACE_TYPE,
)
from mypyc.ops_primitive import binary_ops, unary_ops, func_ops, method_ops, name_ref_ops
from mypyc.ops_list import list_len_op, list_get_item_op, list_set_item_op, new_list_op
from mypyc.ops_dict import new_dict_op, dict_get_item_op
from mypyc.ops_misc import (
    none_op, iter_op, next_op, no_err_occurred_op, py_getattr_op, py_setattr_op,
    fast_isinstance_op,
)
from mypyc.subtype import is_subtype
from mypyc.sametype import is_same_type, is_same_method_signature


def build_ir(modules: List[MypyFile],
             types: Dict[Expression, Type]) -> List[Tuple[str, ModuleIR]]:
    result = []
    mapper = Mapper()

    # Collect all classes defined in the compilation unit.
    classes = []
    for module in modules:
        module_classes = [node for node in module.defs if isinstance(node, ClassDef)]
        classes.extend([(module.fullname(), cdef) for cdef in module_classes])

    # Collect all class mappings so that we can bind arbitrary class name
    # references even if there are import cycles.
    for module_name, cdef in classes:
        class_ir = ClassIR(cdef.name, module_name)
        mapper.type_to_ir[cdef.info] = class_ir

    # Populate structural information in class IR.
    for _, cdef in classes:
        prepare_class_def(cdef, mapper)

    # Generate IR for all modules.
    module_names = [mod.fullname() for mod in modules]
    for module in modules:
        builder = IRBuilder(types, mapper, module_names)
        module.accept(builder)
        ir = ModuleIR(
            builder.imports,
            builder.from_imports,
            mapper.literals,
            builder.functions,
            builder.classes
        )
        result.append((module.fullname(), ir))

    # Compute vtables.
    for _, cdef in classes:
        compute_vtable(mapper.type_to_ir[cdef.info])

    return result


def compute_vtable(cls: ClassIR) -> None:
    """Compute the vtable structure for a class."""
    if cls.vtable is not None: return
    cls.vtable = {}
    entries = cls.vtable_entries
    if cls.base:
        compute_vtable(cls.base)
        assert cls.base.vtable is not None
        cls.vtable.update(cls.base.vtable)
        prefix = cls.base.vtable_entries
    else:
        prefix = []

    # Include the vtable from the parent classes, but handle method overrides.
    for entry in prefix:
        if isinstance(entry, VTableMethod):
            method = entry.method
            if method.name in cls.methods:
                if is_same_method_signature(method.sig, cls.methods[method.name].sig):
                    entry = VTableMethod(cls, cls.methods[method.name])
                else:
                    entry = VTableMethod(cls, cls.glue_methods[(entry.cls, method.name)])
        entries.append(entry)

    for attr in cls.attributes:
        cls.vtable[attr] = len(entries)
        entries.append(VTableAttr(cls, attr, is_getter=True))
        entries.append(VTableAttr(cls, attr, is_getter=False))
    for fn in cls.methods.values():
        cls.vtable[fn.name] = len(entries)
        entries.append(VTableMethod(cls, fn))


class Mapper:
    """Keep track of mappings from mypy concepts to IR concepts.

    This state is shared across all modules in a compilation unit.
    """

    def __init__(self) -> None:
        self.type_to_ir = {}  # type: Dict[TypeInfo, ClassIR]
        # Maps integer, float, and unicode literals to a static name
        self.literals = {}  # type: LiteralsMap

    def type_to_rtype(self, typ: Type) -> RType:
        if isinstance(typ, Instance):
            if typ.type.fullname() == 'builtins.int':
                return int_rprimitive
            elif typ.type.fullname() == 'builtins.float':
                return float_rprimitive
            elif typ.type.fullname() == 'builtins.str':
                return str_rprimitive
            elif typ.type.fullname() == 'builtins.bool':
                return bool_rprimitive
            elif typ.type.fullname() == 'builtins.list':
                return list_rprimitive
            elif typ.type.fullname() == 'builtins.dict':
                return dict_rprimitive
            elif typ.type.fullname() == 'builtins.tuple':
                return tuple_rprimitive  # Varying-length tuple
            elif typ.type in self.type_to_ir:
                return RInstance(self.type_to_ir[typ.type])
            else:
                return object_rprimitive
        elif isinstance(typ, TupleType):
            return RTuple([self.type_to_rtype(t) for t in typ.items])
        elif isinstance(typ, CallableType):
            return object_rprimitive
        elif isinstance(typ, NoneTyp):
            return none_rprimitive
        elif isinstance(typ, UnionType):
            assert len(typ.items) == 2 and any(isinstance(it, NoneTyp) for it in typ.items)
            if isinstance(typ.items[0], NoneTyp):
                value_type = typ.items[1]
            else:
                value_type = typ.items[0]
            return ROptional(self.type_to_rtype(value_type))
        elif isinstance(typ, AnyType):
            return object_rprimitive
        elif isinstance(typ, TypeVarType):
            # Erase type variable to upper bound.
            # TODO: Erase to object if object has value restriction -- or union (once supported)?
            assert not typ.values, 'TypeVar with value restriction not supported'
            return self.type_to_rtype(typ.upper_bound)
        assert False, '%s unsupported' % type(typ)

    def fdef_to_sig(self, fdef: FuncDef) -> FuncSignature:
        assert isinstance(fdef.type, CallableType)
        args = [RuntimeArg(arg.variable.name(), self.type_to_rtype(fdef.type.arg_types[i]))
                for i, arg in enumerate(fdef.arguments)]
        ret = self.type_to_rtype(fdef.type.ret_type)
        return FuncSignature(args, ret)

    def literal_static_name(self, value: Union[int, float, str]) -> str:
        # Include type to distinguish between 1 and 1.0, and so on.
        key = (type(value), value)
        if key not in self.literals:
            if isinstance(value, str):
                prefix = 'unicode_'
            elif isinstance(value, float):
                prefix = 'float_'
            else:
                assert isinstance(value, int)
                prefix = 'int_'
            self.literals[key] = prefix + str(len(self.literals))
        return self.literals[key]


def prepare_class_def(cdef: ClassDef, mapper: Mapper) -> None:
    ir = mapper.type_to_ir[cdef.info]
    info = cdef.info
    for name, node in info.names.items():
        if isinstance(node.node, Var):
            assert node.node.type, "Class member missing type"
            ir.attributes[name] = mapper.type_to_rtype(node.node.type)
        elif isinstance(node.node, FuncDef):
            ir.method_types[name] = mapper.fdef_to_sig(node.node)

    # Set up the parent class
    assert len(info.bases) == 1, "Only single inheritance is supported"
    mro = []
    for cls in info.mro:
        if cls.fullname() == 'builtins.object': continue
        assert cls in mapper.type_to_ir, "Can't subclass cpython types yet"
        mro.append(mapper.type_to_ir[cls])
    if len(mro) > 1:
        ir.base = mro[1]
    ir.mro = mro


class IRBuilder(NodeVisitor[Value]):
    def __init__(self,
                 types: Dict[Expression, Type],
                 mapper: Mapper,
                 modules: List[str]) -> None:
        self.types = types
        self.environment = Environment()
        self.environments = [self.environment]
        self.func_env_classes = []  # type: List[ClassIR]
        self.func_env_class_targets = []  # type: List[AssignmentTarget]
        self.selfs = [Register(none_rprimitive)]  # type: List[Value]
        self.ret_types = []  # type: List[RType]
        self.blocks = []  # type: List[List[BasicBlock]]
        self.functions = []  # type: List[FuncIR]
        self.classes = []  # type: List[ClassIR]
        self.modules = set(modules)

        # These lists operate as stack frames for loops. Each loop adds a new
        # frame (i.e. adds a new empty list [] to the outermost list). Each
        # break or continue is inserted within that frame as they are visited
        # and at the end of the loop the stack is popped and any break/continue
        # gotos have their targets rewritten to the next basic block.
        self.break_gotos = []  # type: List[List[Goto]]
        self.continue_gotos = []  # type: List[List[Goto]]

        self.mapper = mapper
        self.imports = []  # type: List[str]
        self.from_imports = {}  # type: Dict[str, List[Tuple[str, str]]]
        self.imports = []  # type: List[str]

        self.current_module_name = None  # type: Optional[str]

    def visit_mypy_file(self, mypyfile: MypyFile) -> Value:
        if mypyfile.fullname() in ('typing', 'abc'):
            # These module are special; their contents are currently all
            # built-in primitives.
            return INVALID_VALUE

        self.module_name = mypyfile.fullname()

        classes = [node for node in mypyfile.defs if isinstance(node, ClassDef)]

        # Collect all classes.
        for cls in classes:
            ir = self.mapper.type_to_ir[cls.info]
            self.classes.append(ir)

        # Generate ops.
        self.current_module_name = mypyfile.fullname()
        for node in mypyfile.defs:
            node.accept(self)

        return INVALID_VALUE

    def visit_class_def(self, cdef: ClassDef) -> Value:
        ir = self.mapper.type_to_ir[cdef.info]
        for name, node in sorted(cdef.info.names.items(), key=lambda x: x[0]):
            if isinstance(node.node, FuncDef):
                func = self.gen_func_def(node.node, ir.method_sig(node.node.name()), cdef.name)
                self.functions.append(func)
                ir.methods[func.name] = func

                # If this overrides a parent class method with a different type, we need
                # to generate a glue method to mediate between them.
                for cls in ir.mro[1:]:
                    if (name in cls.method_types
                            and not is_same_method_signature(ir.method_types[name],
                                                             cls.method_types[name])):
                        f = self.gen_glue_method(cls.method_types[name], func, ir, cls,
                                                 node.node.line)
                        ir.glue_methods[(cls, name)] = f
                        self.functions.append(f)

        return INVALID_VALUE

    def visit_import(self, node: Import) -> Value:
        if node.is_unreachable or node.is_mypy_only:
            pass
        if not node.is_top_level:
            assert False, "non-toplevel imports not supported"

        for node_id, _ in node.ids:
            self.imports.append(node_id)

        return INVALID_VALUE

    def visit_import_from(self, node: ImportFrom) -> Value:
        if node.is_unreachable or node.is_mypy_only:
            pass

        # TODO support these?
        assert not node.relative

        if node.id not in self.from_imports:
            self.from_imports[node.id] = []

        for name, maybe_as_name in node.names:
            as_name = maybe_as_name or name
            self.from_imports[node.id].append((name, as_name))

        return INVALID_VALUE

    def visit_import_all(self, node: ImportAll) -> Value:
        if node.is_unreachable or node.is_mypy_only:
            pass
        if not node.is_top_level:
            assert False, "non-toplevel imports not supported"

        self.imports.append(node.id)

        return INVALID_VALUE

    def gen_glue_method(self, sig: FuncSignature, target: FuncIR,
                        cls: ClassIR, base: ClassIR, line: int) -> FuncIR:
        """Generate glue methods that mediate between different method types in subclasses.

        For example, if we have:

        class A:
            def f(self, x: int) -> object: ...

        then it is totally permissable to have a subclass

        class B(A):
            def f(self, x: object) -> int: ...

        since '(object) -> int' is a subtype of '(int) -> object' by the usual
        contra/co-variant function subtyping rules.

        The trickiness here is that int and object have different
        runtime representations in mypyc, so A.f and B.f have
        different signatures at the native C level. To deal with this,
        we need to generate glue methods that mediate between the
        different versions by coercing the arguments and return
        values.
        """
        self.enter()

        rt_args = (RuntimeArg(sig.args[0].name, RInstance(cls)),) + sig.args[1:]

        # The environment operates on Vars, so we make some up
        fake_vars = [(Var(arg.name), arg.type) for arg in rt_args]
        args = [self.read_from_target(self.environment.add_local_reg(var, type, is_arg=True), -1)
                for var, type in fake_vars]  # type: List[Value]
        self.ret_types[-1] = sig.ret_type

        arg_types = [arg.type for arg in target.sig.args]
        args = self.coerce_native_call_args(args, arg_types, line)
        retval = self.add(MethodCall(target.ret_type,
                                     args[0],
                                     target.name,
                                     args[1:],
                                     line))
        retval = self.coerce(retval, sig.ret_type, line)
        self.add(Return(retval))

        blocks, env, ret_type = self.leave()
        return FuncIR(target.name + '__' + base.name + '_glue',
                      cls.name, self.module_name,
                      FuncSignature(rt_args, ret_type), blocks, env)

    def gen_func_def(self, fdef: FuncDef, sig: FuncSignature,
                     class_name: Optional[str] = None) -> FuncIR:
        # If there is more than one environment in the environment stack, then we are visiting a
        # non-global function.
        is_nested = len(self.environments) > 1
        # If any of the blocks is another FuncDef, then this contains a nested function.
        contains_nested = any(isinstance(block, FuncDef) for block in fdef.body.body)
        namespace = self.gen_func_ns()

        self.enter(fdef.name())

        if is_nested:
            # Obtain the environment class of the encapsulating function.
            prev_func_env_class = self.func_env_classes[-1]
            prev_func_env_class_target = self.func_env_class_targets[-1]

            # Generate a callable class representing the function. Here, the entirety of the class
            # is not yet generated, but a placeholder reference is needed so that 'self' can have
            # an RInstance type.
            func_callable_class = self.gen_func_callable_class(fdef, namespace,
                                                               prev_func_env_class)

            # Add a 'self' field to the environment, since we instantiate a nested function as a
            # method of a new class representing that original function. Use the callable class as
            # the RInstance type.
            self_target = self.environment.add_local_reg(Var('self'),
                                                         RInstance(func_callable_class),
                                                         is_arg=True)
            self.selfs.append(self.read_from_target(self_target, fdef.line))
        for arg in fdef.arguments:
            assert arg.variable.type, "Function argument missing type"
            self.environment.add_local_reg(arg.variable, self.type_to_rtype(arg.variable.type),
                                           is_arg=True)

        if contains_nested:
            # Add a class representing the environment of the current function. Instantiate it and
            # load it into a register so that inner nested functions can access its members. Note
            # that here, we use a dummy Environment instance, as the entirety of the function's
            # environment has not yet been visited. However, an environment class must be
            # instantiated before nested functions are visited because nested functions need a
            # Value to reference if accessing free variables.
            if is_nested:
                func_env_class = self.gen_func_env_class(fdef, namespace, Environment(),
                                                         prev_func_env_class)
            else:
                func_env_class = self.gen_func_env_class(fdef, namespace, Environment())
            func_env_class_target = self.instantiate_func_env_class(fdef, RInstance(func_env_class), namespace)

        self.ret_types[-1] = sig.ret_type

        fdef.body.accept(self)

        if (is_none_rprimitive(self.ret_types[-1]) or
                is_object_rprimitive(self.ret_types[-1])):
            self.add_implicit_return()
        else:
            self.add_implicit_unreachable()

        blocks, env, ret_type = self.leave()

        if contains_nested:
            compute_vtable(func_env_class)

        if is_nested:
            # Populate the callable class representing the function. Instantiate it and load it
            # into a register in the current environment immediately so that it does not have to be
            # loaded every time the function is called.
            func_callable_class, func = self.populate_func_callable_class(func_callable_class,
                                                                          fdef, blocks, sig, env,
                                                                          prev_func_env_class,
                                                                          prev_func_env_class_target)
            func_callable_class_target = self.instantiate_func_callable_class(fdef, RInstance(func_callable_class), namespace, prev_func_env_class_target)
            compute_vtable(func_callable_class)

            self.selfs.pop()
        else:
            func = FuncIR(fdef.name(), class_name, self.module_name, sig, blocks, env)
        return func

    def visit_func_def(self, fdef: FuncDef) -> Value:
        self.functions.append(self.gen_func_def(fdef, self.mapper.fdef_to_sig(fdef)))
        return INVALID_VALUE

    def add_implicit_return(self) -> None:
        block = self.blocks[-1][-1]
        if not block.ops or not isinstance(block.ops[-1], Return):
            retval = self.add(PrimitiveOp([], none_op, line=-1))
            self.add(Return(retval))

    def add_implicit_unreachable(self) -> None:
        block = self.blocks[-1][-1]
        if not block.ops or not isinstance(block.ops[-1], Return):
            self.add(Unreachable())

    def visit_block(self, block: Block) -> Value:
        for stmt in block.body:
            stmt.accept(self)
        return INVALID_VALUE

    def visit_expression_stmt(self, stmt: ExpressionStmt) -> Value:
        self.accept(stmt.expr)
        return INVALID_VALUE

    def visit_return_stmt(self, stmt: ReturnStmt) -> Value:
        if stmt.expr:
            retval = self.accept(stmt.expr)
            retval = self.coerce(retval, self.ret_types[-1], stmt.line)
        else:
            retval = self.add(PrimitiveOp([], none_op, line=-1))
        self.add(Return(retval))
        return INVALID_VALUE

    def visit_assignment_stmt(self, stmt: AssignmentStmt) -> Value:
        assert len(stmt.lvalues) == 1
        if isinstance(stmt.rvalue, CallExpr) and isinstance(stmt.rvalue.analyzed, TypeVarExpr):
            # Just ignore type variable declarations -- they are a compile-time only thing.
            # TODO: It would be nice to actually construct TypeVar objects to match Python
            #       semantics.
            return INVALID_VALUE
        lvalue = stmt.lvalues[0]
        if stmt.type and isinstance(stmt.rvalue, TempNode):
            # This is actually a variable annotation without initializer. Don't generate
            # an assignment but we need to call get_assignment_target since it adds a
            # name binding as a side effect.
            self.get_assignment_target(lvalue)
            return INVALID_VALUE
        self.assign(lvalue, stmt.rvalue)
        return INVALID_VALUE

    def visit_operator_assignment_stmt(self, stmt: OperatorAssignmentStmt) -> Value:
        target = self.get_assignment_target(stmt.lvalue)
        rreg = self.accept(stmt.rvalue)
        res = self.read_from_target(target, stmt.line)
        res = self.binary_op(res, rreg, stmt.op, stmt.line)
        self.assign_to_target(target, res, res.line)
        return INVALID_VALUE

    def get_assignment_target(self, lvalue: Lvalue) -> AssignmentTarget:
        if isinstance(lvalue, NameExpr):
            # Assign to local variable.
            assert lvalue.kind == LDEF
            assert isinstance(lvalue.node, Var)  # TODO: Can this fail?
            if lvalue.node not in self.environment.symtable:
                # Define a new variable.
                return self.environment.add_local_reg(lvalue.node, self.node_type(lvalue))
            else:
                # Assign to a previously defined variable.
                return self.environment.lookup(lvalue.node)
        elif isinstance(lvalue, IndexExpr):
            # Indexed assignment x[y] = e
            base = self.accept(lvalue.base)
            index = self.accept(lvalue.index)
            return AssignmentTargetIndex(base, index)
        elif isinstance(lvalue, MemberExpr):
            # Attribute assignment x.y = e
            obj = self.accept(lvalue.expr)
            return AssignmentTargetAttr(obj, lvalue.name)

        assert False, 'Unsupported lvalue: %r' % lvalue

    def read_from_target(self, target: AssignmentTarget, line: int) -> Value:
        if isinstance(target, AssignmentTargetRegister):
            return target.register
        if isinstance(target, AssignmentTargetIndex):
            reg = self.translate_special_method_call(target.base,
                                                     '__getitem__',
                                                     [target.index],
                                                     None,
                                                     line)
            if reg is not None:
                return reg
            assert False, target.base.type
        if isinstance(target, AssignmentTargetAttr):
            if isinstance(target.obj.type, RInstance):
                return self.add(GetAttr(target.obj, target.attr, line))
            else:
                return self.py_get_attr(target.obj, target.attr, line)

        assert False, 'Unsupported lvalue: %r' % target

    def assign_to_target(self,
                         target: AssignmentTarget,
                         rvalue_reg: Value,
                         line: int) -> Value:
        if isinstance(target, AssignmentTargetRegister):
            rvalue_reg = self.coerce(rvalue_reg, target.type, line)
            return self.add(Assign(target.register, rvalue_reg))
        elif isinstance(target, AssignmentTargetAttr):
            if isinstance(target.obj_type, RInstance):
                rvalue_reg = self.coerce(rvalue_reg, target.type, line)
                return self.add(SetAttr(target.obj, target.attr, rvalue_reg, line))
            else:
                key = self.load_static_unicode(target.attr)
                boxed_reg = self.box(rvalue_reg)
                return self.add(PrimitiveOp([target.obj, key, boxed_reg], py_setattr_op, line))
        elif isinstance(target, AssignmentTargetIndex):
            target_reg2 = self.translate_special_method_call(
                target.base,
                '__setitem__',
                [target.index, rvalue_reg],
                None,
                line)
            if target_reg2 is not None:
                return target_reg2

            assert False, target.base.type

        assert False, 'Unsupported assignment target'

    def assign(self,
               lvalue: Lvalue,
               rvalue: Expression) -> AssignmentTarget:
        target = self.get_assignment_target(lvalue)
        rvalue_reg = self.accept(rvalue)
        self.assign_to_target(target, rvalue_reg, rvalue.line)
        return target

    def visit_if_stmt(self, stmt: IfStmt) -> Value:
        # If statements are normalized
        assert len(stmt.expr) == 1

        branches = self.process_conditional(stmt.expr[0])
        if_body = self.new_block()
        self.set_branches(branches, True, if_body)
        stmt.body[0].accept(self)
        if_leave = self.add_leave()
        if stmt.else_body:
            else_body = self.new_block()
            self.set_branches(branches, False, else_body)
            stmt.else_body.accept(self)
            else_leave = self.add_leave()
            next = self.new_block()
            if else_leave:
                else_leave.label = next.label
        else:
            # No else block.
            next = self.new_block()
            self.set_branches(branches, False, next)
        if if_leave:
            if_leave.label = next.label
        return INVALID_VALUE

    def add_leave(self) -> Optional[Goto]:
        if not self.blocks[-1][-1].ops or not isinstance(self.blocks[-1][-1].ops[-1], Return):
            leave = Goto(INVALID_LABEL)
            self.add(leave)
            return leave
        return None

    def push_loop_stack(self) -> None:
        self.break_gotos.append([])
        self.continue_gotos.append([])

    def pop_loop_stack(self, continue_block: BasicBlock, break_block: BasicBlock) -> None:
        for continue_goto in self.continue_gotos.pop():
            continue_goto.label = continue_block.label

        for break_goto in self.break_gotos.pop():
            break_goto.label = break_block.label

    def visit_while_stmt(self, s: WhileStmt) -> Value:
        self.push_loop_stack()

        # Split block so that we get a handle to the top of the loop.
        goto = Goto(INVALID_LABEL)
        self.add(goto)
        top = self.new_block()
        goto.label = top.label
        branches = self.process_conditional(s.expr)

        body = self.new_block()
        # Bind "true" branches to the body block.
        self.set_branches(branches, True, body)
        s.body.accept(self)
        # Add branch to the top at the end of the body.
        self.add(Goto(top.label))
        next = self.new_block()
        # Bind "false" branches to the new block.
        self.set_branches(branches, False, next)

        self.pop_loop_stack(top, next)
        return INVALID_VALUE

    def visit_for_stmt(self, s: ForStmt) -> Value:
        if (isinstance(s.expr, CallExpr)
                and isinstance(s.expr.callee, RefExpr)
                and s.expr.callee.fullname == 'builtins.range'):
            self.push_loop_stack()

            # Special case for x in range(...)
            # TODO: Check argument counts and kinds; check the lvalue
            end = s.expr.args[0]
            end_reg = self.accept(end)

            # Initialize loop index to 0.
            assign_target = self.assign(s.index, IntExpr(0))
            assert isinstance(assign_target, AssignmentTargetRegister)
            index_reg = assign_target.register
            goto = Goto(INVALID_LABEL)
            self.add(goto)

            # Add loop condition check.
            top = self.new_block()
            goto.label = top.label
            comparison = self.binary_op(index_reg, end_reg, '<', s.line)
            branch = Branch(comparison, INVALID_LABEL, INVALID_LABEL, Branch.BOOL_EXPR)
            self.add(branch)
            branches = [branch]

            body = self.new_block()
            self.set_branches(branches, True, body)
            s.body.accept(self)

            end_goto = Goto(INVALID_LABEL)
            self.add(end_goto)
            end_block = self.new_block()
            end_goto.label = end_block.label

            # Increment index register.
            one_reg = self.add(LoadInt(1))
            self.add(Assign(index_reg, self.binary_op(index_reg, one_reg, '+', s.line)))

            # Go back to loop condition check.
            self.add(Goto(top.label))
            next = self.new_block()
            self.set_branches(branches, False, next)

            self.pop_loop_stack(end_block, next)
            return INVALID_VALUE

        elif is_list_rprimitive(self.node_type(s.expr)):
            self.push_loop_stack()

            expr_reg = self.accept(s.expr)

            index_reg = self.alloc_temp(int_rprimitive)
            self.add(Assign(index_reg, self.add(LoadInt(0))))

            one_reg = self.add(LoadInt(1))

            assert isinstance(s.index, NameExpr)
            assert isinstance(s.index.node, Var)
            lvalue = self.environment.add_local_reg(s.index.node, self.node_type(s.index))

            condition_block = self.goto_new_block()

            # For compatibility with python semantics we recalculate the length
            # at every iteration.
            len_reg = self.add(PrimitiveOp([expr_reg], list_len_op, s.line))

            comparison = self.binary_op(index_reg, len_reg, '<', s.line)
            branch = Branch(comparison, INVALID_LABEL, INVALID_LABEL, Branch.BOOL_EXPR)
            self.add(branch)
            branches = [branch]

            body_block = self.new_block()
            self.set_branches(branches, True, body_block)

            target_list_type = self.types[s.expr]
            assert isinstance(target_list_type, Instance)
            target_type = self.type_to_rtype(target_list_type.args[0])
            value_box = self.add(PrimitiveOp([expr_reg, index_reg], list_get_item_op, s.line))

            self.assign_to_target(lvalue,
                                  self.unbox_or_cast(value_box, target_type, s.line), s.line)

            s.body.accept(self)

            end_block = self.goto_new_block()
            self.add(Assign(index_reg, self.binary_op(index_reg, one_reg, '+', s.line)))
            self.add(Goto(condition_block.label))

            next_block = self.new_block()
            self.set_branches(branches, False, next_block)

            self.pop_loop_stack(end_block, next_block)

            return INVALID_VALUE

        else:
            self.push_loop_stack()

            assert isinstance(s.index, NameExpr)
            assert isinstance(s.index.node, Var)
            lvalue = self.environment.add_local_reg(s.index.node, object_rprimitive)

            # Define registers to contain the expression, along with the iterator that will be used
            # for the for-loop.
            expr_reg = self.accept(s.expr)
            iter_reg = self.add(PrimitiveOp([expr_reg], iter_op, s.line))

            # Create a block for where the __next__ function will be called on the iterator and
            # checked to see if the value returned is NULL, which would signal either the end of
            # the Iterable being traversed or an exception being raised. Note that Branch.IS_ERROR
            # checks only for NULL (an exception does not necessarily have to be raised).
            next_block = self.goto_new_block()
            next_reg = self.add(PrimitiveOp([iter_reg], next_op, s.line))
            branch = Branch(next_reg, INVALID_LABEL, INVALID_LABEL, Branch.IS_ERROR)
            self.add(branch)
            branches = [branch]

            # Create a new block for the body of the loop. Set the previous branch to go here if
            # the conditional evaluates to false. Assign the value obtained from __next__ to the
            # lvalue so that it can be referenced by code in the body of the loop. At the end of
            # the body, goto the label that calls the iterator's __next__ function again.
            body_block = self.new_block()
            self.set_branches(branches, False, body_block)
            self.assign_to_target(lvalue, next_reg, s.line)
            s.body.accept(self)
            self.add(Goto(next_block.label))

            # Create a new block for when the loop is finished. Set the branch to go here if the
            # conditional evaluates to true. If an exception was raised during the loop, then
            # err_reg wil be set to True. If no_err_occurred_op returns False, then the exception
            # will be propagated using the ERR_FALSE flag.
            end_block = self.new_block()
            self.set_branches(branches, True, end_block)
            self.add(PrimitiveOp([], no_err_occurred_op, s.line))

            self.pop_loop_stack(next_block, end_block)

            return INVALID_VALUE

    def visit_break_stmt(self, node: BreakStmt) -> Value:
        self.break_gotos[-1].append(Goto(INVALID_LABEL))
        self.add(self.break_gotos[-1][-1])
        self.new_block()
        return INVALID_VALUE

    def visit_continue_stmt(self, node: ContinueStmt) -> Value:
        self.continue_gotos[-1].append(Goto(INVALID_LABEL))
        self.add(self.continue_gotos[-1][-1])
        self.new_block()
        return INVALID_VALUE

    def visit_unary_expr(self, expr: UnaryExpr) -> Value:
        ereg = self.accept(expr.expr)
        ops = unary_ops.get(expr.op, [])
        target = self.matching_primitive_op(ops, [ereg], expr.line)
        assert target, 'Unsupported unary operation: %s' % expr.op
        return target

    def visit_op_expr(self, expr: OpExpr) -> Value:
        return self.binary_op(self.accept(expr.left), self.accept(expr.right), expr.op, expr.line)

    def matching_primitive_op(self,
                              candidates: List[OpDescription],
                              args: List[Value],
                              line: int,
                              result_type: Optional[RType] = None) -> Optional[Value]:
        # Find the highest-priority primitive op that matches.
        matching = None  # type: Optional[OpDescription]
        for desc in candidates:
            if len(desc.arg_types) != len(args):
                continue
            if all(is_subtype(actual.type, formal)
                   for actual, formal in zip(args, desc.arg_types)):
                if matching:
                    assert matching.priority != desc.priority, 'Ambiguous:\n1) %s\n2) %s' % (
                        matching, desc)
                    if desc.priority > matching.priority:
                        matching = desc
                else:
                    matching = desc
        if matching:
            target = self.primitive_op(matching, args, line)
            if result_type and not is_same_type(target.type, result_type):
                if is_none_rprimitive(result_type):
                    # Special case None return. The actual result may actually be a bool
                    # and so we can't just coerce it.
                    target = self.add(PrimitiveOp([], none_op, line))
                else:
                    target = self.coerce(target, result_type, line)
            return target
        return None

    def binary_op(self,
                  lreg: Value,
                  rreg: Value,
                  expr_op: str,
                  line: int) -> Value:
        ops = binary_ops.get(expr_op, [])
        target = self.matching_primitive_op(ops, [lreg, rreg], line)
        assert target, 'Unsupported binary operation: %s' % expr_op
        return target

    def visit_index_expr(self, expr: IndexExpr) -> Value:
        base = self.accept(expr.base)

        if isinstance(base.type, RTuple):
            assert isinstance(expr.index, IntExpr)  # TODO
            return self.add(TupleGet(base, expr.index.value, expr.line))

        index_reg = self.accept(expr.index)
        target_reg = self.translate_special_method_call(
            base,
            '__getitem__',
            [index_reg],
            self.node_type(expr),
            expr.line)
        if target_reg is not None:
            return target_reg

        assert False, 'Unsupported indexing operation'

    def visit_int_expr(self, expr: IntExpr) -> Value:
        if expr.value > MAX_SHORT_INT:
            return self.load_static_int(expr.value)
        return self.add(LoadInt(expr.value))

    def visit_float_expr(self, expr: FloatExpr) -> Value:
        return self.load_static_float(expr.value)

    def is_native_name_expr(self, expr: NameExpr) -> bool:
        assert expr.node, "RefExpr not resolved"
        if '.' in expr.node.fullname():
            module_name = '.'.join(expr.node.fullname().split('.')[:-1])
            return module_name in self.modules

        return True

    def is_native_module_name_expr(self, expr: NameExpr) -> bool:
        return self.is_native_name_expr(expr) and expr.kind == GDEF

    def visit_name_expr(self, expr: NameExpr) -> Value:
        assert expr.node, "RefExpr not resolved"
        fullname = expr.node.fullname()
        if fullname in name_ref_ops:
            # Use special access op for this particular name.
            desc = name_ref_ops[fullname]
            assert desc.result_type is not None
            return self.add(PrimitiveOp([], desc, expr.line))

        # TODO: Behavior currently only defined for Var and FuncDef node types.
        if expr.kind == LDEF:
            try:
                target = self.environment.lookup(expr.node)
                return self.read_from_target(target, expr.line)
            except KeyError:
                # If there is a KeyError, then that means the target could not be found in the
                # current scope. We need to search the hierarchy of environments to check if the
                # target was defined in an outer scope.
                symbol_found = False
                index = len(self.environments) - 2
                while not symbol_found and index >= 1:
                    # At each level, first check the function environment class. If it cannot be
                    # found there, check the symbol table in the environment stack.
                    if expr.node.name() in self.func_env_classes[index - 1].attributes:
                        rtype = self.func_env_classes[index - 1].attributes[expr.node.name()]
                        symbol_found = True
                    else:
                        symtable = self.environments[index].symtable
                        if expr.node in symtable:
                            rtype = symtable[expr.node].type
                            symbol_found = True
                    index -= 1

                # Now, add the symbol to the function environment class corresponding to where it
                # was defined. Then use GetAttr to retrieve the Value associated with the symbol.
                if symbol_found:
                    self.func_env_classes[index].attributes[expr.node.name()] = rtype
                    env = self.selfs[-1]
                    for i in range(index, len(self.func_env_classes)):
                        env = self.add(GetAttr(env, 'env', expr.line))
                    val = self.add(GetAttr(env, expr.node.name(), expr.line))
                    # TODO: Figure out how to cache the value into a register so that it doesn't
                    # need to be loaded next time. Perhaps put it into the environment symbol
                    # table?
                    return val

                assert False, 'expression {} not defined in current scope'.format(expr.name)
        else:
            return self.load_global(expr)

    def is_module_member_expr(self, expr: MemberExpr) -> bool:
        return isinstance(expr.expr, RefExpr) and expr.expr.kind == MODULE_REF

    def visit_member_expr(self, expr: MemberExpr) -> Value:
        if self.is_module_member_expr(expr):
            return self.load_static_module_attr(expr)
        else:
            obj = self.accept(expr.expr)
            if isinstance(obj.type, RInstance):
                return self.add(GetAttr(obj, expr.name, expr.line))
            else:
                return self.py_get_attr(obj, expr.name, expr.line)

    def py_get_attr(self, obj: Value, attr: str, line: int) -> Value:
        key = self.load_static_unicode(attr)
        return self.add(PrimitiveOp([obj, key], py_getattr_op, line))

    def py_call(self, function: Value, args: List[Value],
                target_type: RType, line: int) -> Value:
        arg_boxes = [self.box(arg) for arg in args]  # type: List[Value]
        return self.add(PyCall(function, arg_boxes, line))

    def py_method_call(self,
                       obj: Value,
                       method: Value,
                       args: List[Value],
                       target_type: RType,
                       line: int) -> Value:
        arg_boxes = [self.box(arg) for arg in args]  # type: List[Value]
        return self.add(PyMethodCall(obj, method, arg_boxes))

    def coerce_native_call_args(self,
                                args: Sequence[Value],
                                arg_types: Sequence[RType],
                                line: int) -> List[Value]:
        coerced_arg_regs = []
        for reg, arg_type in zip(args, arg_types):
            coerced_arg_regs.append(self.coerce(reg, arg_type, line))
        return coerced_arg_regs

    def visit_call_expr(self, expr: CallExpr) -> Value:
        if isinstance(expr.analyzed, CastExpr):
            return self.translate_cast_expr(expr.analyzed)

        callee = expr.callee
        if isinstance(callee, IndexExpr) and isinstance(callee.analyzed, TypeApplication):
            callee = callee.analyzed.expr  # Unwrap type application

        if isinstance(callee, MemberExpr):
            # TODO: Could be call to module-level function
            return self.translate_method_call(expr, callee)
        else:
            return self.translate_call(expr, callee)

    def translate_call(self, expr: CallExpr, callee: Expression) -> Value:
        """Translate a non-method call."""
        assert isinstance(callee, NameExpr)  # TODO: Allow arbitrary callees

        # Gen the args
        fullname = callee.fullname
        args = [self.accept(arg) for arg in expr.args]

        if fullname == 'builtins.len' and len(expr.args) == 1 and expr.arg_kinds == [ARG_POS]:
            expr_rtype = args[0].type
            if isinstance(expr_rtype, RTuple):
                # len() of fixed-length tuple can be trivially determined statically.
                return self.add(LoadInt(len(expr_rtype.types)))
        if (fullname == 'builtins.isinstance'
                and len(expr.args) == 2
                and expr.arg_kinds == [ARG_POS, ARG_POS]
                and isinstance(expr.args[1], NameExpr)
                and isinstance(expr.args[1].node, TypeInfo)
                and self.is_native_module_name_expr(expr.args[1])):
            # Special case native isinstance() checks as this makes them much faster.
            return self.primitive_op(fast_isinstance_op, args, expr.line)

        # Handle data-driven special-cased primitive call ops.
        target_type = self.node_type(expr)
        if fullname is not None and expr.arg_kinds == [ARG_POS] * len(args):
            ops = func_ops.get(fullname, [])
            target = self.matching_primitive_op(ops, args, expr.line)
            if target:
                return target

        fn = callee.fullname
        # Try to generate a native call. Don't rely on the inferred callee
        # type, since it may have type variable substitutions that aren't
        # valid at runtime (due to type erasure). Instead pick the declared
        # signature of the native function as the true signature.
        signature = self.get_native_signature(callee)
        if signature and fn:
            # Native call
            arg_types = [self.type_to_rtype(arg_type) for arg_type in signature.arg_types]
            args = self.coerce_native_call_args(args, arg_types, expr.line)
            ret_type = self.type_to_rtype(signature.ret_type)
            return self.add(Call(ret_type, fn, args, expr.line))
        else:
            # Fall back to a Python call
            function = self.accept(callee)
            return self.py_call(function, args, target_type, expr.line)

    def get_native_signature(self, callee: NameExpr) -> Optional[CallableType]:
        """Get the signature of a native function, or return None if not available.

        This only works for normal functions, not methods.
        """
        signature = None
        if self.is_native_module_name_expr(callee):
            node = callee.node
            if isinstance(node, TypeInfo):
                node = node['__init__'].node
                if isinstance(node, FuncDef) and isinstance(node.type, CallableType):
                    signature = bind_self(node.type)
                    # "__init__" has None return, but the type object returns
                    # in instance.  Take the instance return type from the
                    # inferred callee type, which we can trust since it can't
                    # be erased from a type variable.
                    inferred_sig = self.types[callee]
                    assert isinstance(inferred_sig, CallableType)
                    signature = signature.copy_modified(ret_type=inferred_sig.ret_type)
            elif isinstance(node, FuncDef) and isinstance(node.type, CallableType):
                signature = node.type
        return signature

    def translate_method_call(self, expr: CallExpr, callee: MemberExpr) -> Value:
        if self.is_module_member_expr(callee):
            # Fall back to a PyCall for module calls
            function = self.accept(callee)
            args = [self.accept(arg) for arg in expr.args]
            return self.py_call(function, args, self.node_type(expr), expr.line)
        else:
            obj = self.accept(callee.expr)
            args = [self.accept(arg) for arg in expr.args]
            assert callee.expr in self.types
            receiver_rtype = self.node_type(callee.expr)

            # First try to do a special-cased method call
            target = self.translate_special_method_call(
                obj, callee.name, args, self.node_type(expr), expr.line)
            if target:
                return target

            # If the base type is one of ours, do a MethodCall
            if isinstance(receiver_rtype, RInstance):
                # Look up the declared signature of the method, since the
                # inferred signature can have type variable substitutions which
                # aren't valid at runtime due to type erasure.
                typ = self.types[callee.expr]
                assert isinstance(typ, Instance)
                method = typ.type.get(callee.name)
                if method and isinstance(method.node, FuncDef) and isinstance(method.node.type,
                                                                              CallableType):
                    sig = method.node.type
                    arg_types = [self.type_to_rtype(arg_type)
                                 for arg_type in sig.arg_types[1:]]
                    arg_regs = self.coerce_native_call_args(args, arg_types, expr.line)
                    target_type = self.type_to_rtype(sig.ret_type)
                    return self.add(MethodCall(target_type, obj, callee.name,
                                               arg_regs, expr.line))

            # Fall back to Python method call
            method_name = self.load_static_unicode(callee.name)
            return self.py_method_call(
                obj, method_name, args, self.node_type(expr), expr.line)

    def translate_cast_expr(self, expr: CastExpr) -> Value:
        src = self.accept(expr.expr)
        target_type = self.type_to_rtype(expr.type)
        return self.coerce(src, target_type, expr.line)

    def visit_conditional_expr(self, expr: ConditionalExpr) -> Value:
        branches = self.process_conditional(expr.cond)
        expr_type = self.node_type(expr)
        # Having actual Phi nodes would be really nice here!
        target = self.alloc_temp(expr_type)

        if_body = self.new_block()
        self.set_branches(branches, True, if_body)
        true_value = self.accept(expr.if_expr)
        true_value = self.coerce(true_value, expr_type, expr.line)
        self.add(Assign(target, true_value))
        if_goto_next = Goto(INVALID_LABEL)
        self.add(if_goto_next)

        else_body = self.new_block()
        self.set_branches(branches, False, else_body)
        false_value = self.accept(expr.else_expr)
        false_value = self.coerce(false_value, expr_type, expr.line)
        self.add(Assign(target, false_value))
        else_goto_next = Goto(INVALID_LABEL)
        self.add(else_goto_next)

        next = self.new_block()
        if_goto_next.label = next.label
        else_goto_next.label = next.label

        return target

    def translate_special_method_call(self,
                                      base_reg: Value,
                                      name: str,
                                      args: List[Value],
                                      result_type: Optional[RType],
                                      line: int) -> Optional[Value]:
        """Translate a method call which is handled nongenerically.

        These are special in the sense that we have code generated specifically for them.
        They tend to be method calls which have equivalents in C that are more direct
        than calling with the PyObject api.

        Return None if no translation found; otherwise return the target register.
        """
        ops = method_ops.get(name, [])
        return self.matching_primitive_op(ops, [base_reg] + args, line, result_type=result_type)

    def visit_list_expr(self, expr: ListExpr) -> Value:
        items = [self.accept(item) for item in expr.items]
        return self.primitive_op(new_list_op, items, expr.line)

    def visit_tuple_expr(self, expr: TupleExpr) -> Value:
        tuple_type = self.node_type(expr)
        assert isinstance(tuple_type, RTuple)

        items = []
        for item_expr, item_type in zip(expr.items, tuple_type.types):
            reg = self.accept(item_expr)
            items.append(self.coerce(reg, item_type, item_expr.line))
        return self.add(TupleSet(items, expr.line))

    def visit_dict_expr(self, expr: DictExpr) -> Value:
        assert not expr.items  # TODO
        return self.add(PrimitiveOp([], new_dict_op, expr.line))

    def visit_str_expr(self, expr: StrExpr) -> Value:
        return self.load_static_unicode(expr.value)

    # Conditional expressions

    def process_conditional(self, e: Node) -> List[Branch]:
        if isinstance(e, ComparisonExpr):
            return self.process_comparison(e)
        elif isinstance(e, OpExpr) and e.op in ['and', 'or']:
            if e.op == 'and':
                # Short circuit 'and' in a conditional context.
                lbranches = self.process_conditional(e.left)
                new = self.new_block()
                self.set_branches(lbranches, True, new)
                rbranches = self.process_conditional(e.right)
                return lbranches + rbranches
            else:
                # Short circuit 'or' in a conditional context.
                lbranches = self.process_conditional(e.left)
                new = self.new_block()
                self.set_branches(lbranches, False, new)
                rbranches = self.process_conditional(e.right)
                return lbranches + rbranches
        elif isinstance(e, UnaryExpr) and e.op == 'not':
            branches = self.process_conditional(e.expr)
            for b in branches:
                b.invert()
            return branches
        # Catch-all for arbitrary expressions.
        else:
            reg = self.accept(e)
            branch = Branch(reg, INVALID_LABEL, INVALID_LABEL, Branch.BOOL_EXPR)
            self.add(branch)
            return [branch]

    def process_comparison(self, e: ComparisonExpr) -> List[Branch]:
        # TODO: Verify operand types.
        assert len(e.operators) == 1, 'more than 1 operator not supported'
        op = e.operators[0]
        if op in ['is', 'is not']:
            # Special case 'is' checks.
            # TODO: check if right operand is None
            left = self.accept(e.operands[0])
            branch = Branch(left, INVALID_LABEL, INVALID_LABEL,
                            Branch.IS_NONE)
            if op == 'is not':
                branch.negated = True
        else:
            # General comparison -- evaluate both operands.
            left = self.accept(e.operands[0])
            right = self.accept(e.operands[1])
            # Generate a bool value and branch based on it.
            if op in ['in', 'not in']:
                target = self.binary_op(left, right, 'in', e.line)
            else:
                target = self.binary_op(left, right, op, e.line)
            target = self.coerce(target, bool_rprimitive, e.line)
            branch = Branch(target, INVALID_LABEL, INVALID_LABEL,
                            Branch.BOOL_EXPR)
            if op == 'not in':
                branch.negated = True
        self.add(branch)
        return [branch]

    def set_branches(self, branches: List[Branch], condition: bool,
                     target: BasicBlock) -> None:
        """Set branch targets for the given condition (True or False).

        If the target has already been set for a branch, skip the branch.
        """
        for b in branches:
            if condition:
                if b.true < 0:
                    b.true = target.label
            else:
                if b.false < 0:
                    b.false = target.label

    def visit_pass_stmt(self, o: PassStmt) -> Value:
        return INVALID_VALUE

    def visit_cast_expr(self, o: CastExpr) -> Value:
        assert False, "CastExpr handled in CallExpr"

    # Unimplemented constructs
    # TODO: some of these are actually things that should never show up,
    # so properly sort those out.
    def visit__promote_expr(self, o: PromoteExpr) -> Value:
        raise NotImplementedError

    def visit_await_expr(self, o: AwaitExpr) -> Value:
        raise NotImplementedError

    def visit_backquote_expr(self, o: BackquoteExpr) -> Value:
        raise NotImplementedError

    def visit_assert_stmt(self, o: AssertStmt) -> Value:
        raise NotImplementedError

    def visit_bytes_expr(self, o: BytesExpr) -> Value:
        raise NotImplementedError

    def visit_comparison_expr(self, o: ComparisonExpr) -> Value:
        raise NotImplementedError

    def visit_complex_expr(self, o: ComplexExpr) -> Value:
        raise NotImplementedError

    def visit_decorator(self, o: Decorator) -> Value:
        raise NotImplementedError

    def visit_del_stmt(self, o: DelStmt) -> Value:
        raise NotImplementedError

    def visit_dictionary_comprehension(self, o: DictionaryComprehension) -> Value:
        raise NotImplementedError

    def visit_ellipsis(self, o: EllipsisExpr) -> Value:
        raise NotImplementedError

    def visit_enum_call_expr(self, o: EnumCallExpr) -> Value:
        raise NotImplementedError

    def visit_exec_stmt(self, o: ExecStmt) -> Value:
        raise NotImplementedError

    def visit_generator_expr(self, o: GeneratorExpr) -> Value:
        raise NotImplementedError

    def visit_global_decl(self, o: GlobalDecl) -> Value:
        raise NotImplementedError

    def visit_lambda_expr(self, o: LambdaExpr) -> Value:
        raise NotImplementedError

    def visit_list_comprehension(self, o: ListComprehension) -> Value:
        raise NotImplementedError

    def visit_set_comprehension(self, o: SetComprehension) -> Value:
        raise NotImplementedError

    def visit_namedtuple_expr(self, o: NamedTupleExpr) -> Value:
        raise NotImplementedError

    def visit_newtype_expr(self, o: NewTypeExpr) -> Value:
        raise NotImplementedError

    def visit_nonlocal_decl(self, o: NonlocalDecl) -> Value:
        raise NotImplementedError

    def visit_overloaded_func_def(self, o: OverloadedFuncDef) -> Value:
        raise NotImplementedError

    def visit_print_stmt(self, o: PrintStmt) -> Value:
        raise NotImplementedError

    def visit_raise_stmt(self, o: RaiseStmt) -> Value:
        raise NotImplementedError

    def visit_reveal_expr(self, o: RevealExpr) -> Value:
        raise NotImplementedError

    def visit_set_expr(self, o: SetExpr) -> Value:
        raise NotImplementedError

    def visit_slice_expr(self, o: SliceExpr) -> Value:
        raise NotImplementedError

    def visit_star_expr(self, o: StarExpr) -> Value:
        raise NotImplementedError

    def visit_super_expr(self, o: SuperExpr) -> Value:
        raise NotImplementedError

    def visit_temp_node(self, o: TempNode) -> Value:
        raise NotImplementedError

    def visit_try_stmt(self, o: TryStmt) -> Value:
        raise NotImplementedError

    def visit_type_alias_expr(self, o: TypeAliasExpr) -> Value:
        raise NotImplementedError

    def visit_type_application(self, o: TypeApplication) -> Value:
        raise NotImplementedError

    def visit_type_var_expr(self, o: TypeVarExpr) -> Value:
        raise NotImplementedError

    def visit_typeddict_expr(self, o: TypedDictExpr) -> Value:
        raise NotImplementedError

    def visit_unicode_expr(self, o: UnicodeExpr) -> Value:
        raise NotImplementedError

    def visit_var(self, o: Var) -> Value:
        raise NotImplementedError

    def visit_with_stmt(self, o: WithStmt) -> Value:
        raise NotImplementedError

    def visit_yield_from_expr(self, o: YieldFromExpr) -> Value:
        raise NotImplementedError

    def visit_yield_expr(self, o: YieldExpr) -> Value:
        raise NotImplementedError

    # Helpers

    def enter(self, name: Optional[str] = None) -> None:
        self.environment = Environment(name)
        self.environments.append(self.environment)
        self.ret_types.append(none_rprimitive)
        self.blocks.append([])
        self.new_block()

    def new_block(self) -> BasicBlock:
        new = BasicBlock(Label(len(self.blocks[-1])))
        self.blocks[-1].append(new)
        return new

    def goto_new_block(self) -> BasicBlock:
        goto = Goto(INVALID_LABEL)
        self.add(goto)
        block = self.new_block()
        goto.label = block.label
        return block

    def leave(self) -> Tuple[List[BasicBlock], Environment, RType]:
        blocks = self.blocks.pop()
        env = self.environments.pop()
        ret_type = self.ret_types.pop()
        self.environment = self.environments[-1]
        return blocks, env, ret_type

    def add(self, op: Op) -> Value:
        self.blocks[-1][-1].ops.append(op)
        if isinstance(op, RegisterOp):
            self.environment.add_op(op)
        return op

    def primitive_op(self, desc: OpDescription, args: List[Value], line: int) -> Value:
        assert desc.result_type is not None
        coerced = []
        for i, arg in enumerate(args):
            formal_type = self.op_arg_type(desc, i)
            arg = self.coerce(arg, formal_type, line)
            coerced.append(arg)
        target = self.add(PrimitiveOp(coerced, desc, line))
        return target

    def op_arg_type(self, desc: OpDescription, n: int) -> RType:
        if n >= len(desc.arg_types):
            assert desc.is_var_arg
            return desc.arg_types[-1]
        return desc.arg_types[n]

    def accept(self, node: Node) -> Value:
        res = node.accept(self)
        if isinstance(node, Expression):
            assert res != INVALID_VALUE
            res = self.coerce(res, self.node_type(node), node.line)
        return res

    def alloc_temp(self, type: RType) -> Register:
        return self.environment.add_temp(type)

    def type_to_rtype(self, typ: Type) -> RType:
        return self.mapper.type_to_rtype(typ)

    def node_type(self, node: Expression) -> RType:
        if isinstance(node, IntExpr):
            # TODO: Don't special case IntExpr
            return int_rprimitive
        mypy_type = self.types[node]
        return self.type_to_rtype(mypy_type)

    def box(self, src: Value) -> Value:
        if src.type.is_unboxed:
            return self.add(Box(src))
        else:
            return src

    def unbox_or_cast(self, src: Value, target_type: RType, line: int) -> Value:
        if target_type.is_unboxed:
            return self.add(Unbox(src, target_type, line))
        else:
            return self.add(Cast(src, target_type, line))

    def box_expr(self, expr: Expression) -> Value:
        return self.box(self.accept(expr))

    def gen_func_ns(self) -> str:
        """Generates a namespace for a nested function using its outer function names."""
        return '_'.join(env.name for env in self.environments if env.name)

    def gen_func_env_class(self,
                           fdef: FuncDef,
                           namespace: str,
                           env: Environment,
                           prev_env_class: Optional[ClassIR] = None) -> ClassIR:
        """Generates a class representing a function environment."""
        ir = ClassIR(name='{}_{}_env'.format(fdef.name(), namespace),
                     module_name=self.module_name)
        for symbol, target in env.symtable.items():
            ir.attributes[symbol.name()] = target.type
        if prev_env_class:
            ir.attributes['prev_env'] = RInstance(prev_env_class)
        ir.mro = [ir]
        self.func_env_classes.append(ir)
        self.classes.append(ir)
        return ir

    def gen_func_callable_class(self,
                                fdef: FuncDef,
                                namespace: str,
                                env_class: ClassIR) -> ClassIR:
        name = '{}_{}_obj'.format(fdef.name(), namespace)
        ir = ClassIR(name, self.module_name)
        ir.attributes['env'] = RInstance(env_class)
        ir.mro = [ir]
        self.classes.append(ir)
        return ir

    def populate_func_callable_class(self,
                                     cir: ClassIR,
                                     fdef: FuncDef,
                                     blocks: List[BasicBlock],
                                     sig: FuncSignature,
                                     env: Environment,
                                     env_class: ClassIR,
                                     env_class_target: AssignmentTarget) -> Tuple[ClassIR, FuncIR]:
        """Generates a callable class representing a nested function.

        This takes a FuncDef and its associated namespace, blocks, environment, and return type and
        builds a ClassIR with its '__call__' method implemented to represent the function. Note
        that the name of the function is changed to be '__call__', and a 'self' parameter is added
        to its list of arguments, as it becomes a class method. The name of the newly constructed
        class is generated using the names of the functions that enclose the given nested function.

        Returns a newly constructed FuncIR associated with the given FuncDef.
        """
        # Construct blocks for __init__ function.
        self.enter()
        self_arg = RuntimeArg('self', object_rprimitive)
        self_target = self.environment.add_local_reg(Var('self'),
                                                         RInstance(cir),
                                                         is_arg=True)

        env_reg = self.instantiate_func_env_class(fdef, RInstance(env_class), 'foo')

        self.add(SetAttr(self_target, 'env', env_class_target, fdef.line))
        self.add_implicit_return()
        init_blocks, init_env, init_ret_type = self.leave()

        init_func_sig = FuncSignature((self_arg,), none_rprimitive)
        init_func_ir = FuncIR('__init__', cir.name, self.module_name, init_func_sig, init_blocks, init_env)
        cir.methods['__init__'] = init_func_ir
        self.functions.append(init_func_ir)

        call_func_sig = FuncSignature((self_arg,) + sig.args, sig.ret_type)
        call_func_ir = FuncIR('__call__', cir.name, self.module_name, call_func_sig, blocks, env)
        cir.methods['__call__'] = call_func_ir
        return cir, call_func_ir

    def instantiate_func_env_class(self,
                                   fdef: FuncDef,
                                   ret_type: RType,
                                   namespace: str,
                                   prev_env_class: Optional[AssignmentTarget] = None) -> AssignmentTargetRegister:
        """Assigns an environment class to a register named after the given function definition."""
        fullname = '{}.{}_{}_env'.format(self.module_name, fdef.name(), namespace)
        env_name = '{}_env'.format(fdef.name())
        env_reg = self.add(Call(ret_type, fullname, [], fdef.line))
        if prev_env_class:
            self.add(SetAttr(env_reg, 'prev_env', prev_env_class, fdef.line))
        self.func_env_class_targets.append(env_reg)
        return env_reg

        # env_target = self.environment.add_local_reg(Var(env_name), ret_type)
        # self.assign_to_target(env_target, env_reg, fdef.line)
        # self.func_env_class_targets.append(env_target)
        # return env_target

    def instantiate_func_callable_class(self,
                                        fdef: FuncDef,
                                        ret_type: RType,
                                        namespace: str,
                                        env_class: AssignmentTarget) -> AssignmentTargetRegister:
        """Assigns a callable class to a register named after the given function definition."""
        fullname = '{}.{}_{}_obj'.format(self.module_name, fdef.name(), namespace)
        func_reg = self.add(Call(ret_type, fullname, [], fdef.line))
        func_target = self.environment.add_local_reg(fdef, ret_type)
        self.assign_to_target(func_target, func_reg, fdef.line)
        # Set the environment field of the function's callable class to point at the given
        # assignment target.
        # TODO: Instead move this logic into an __init__ function of the actual environment class.
        self.add(SetAttr(func_target, 'env', env_class, fdef.line))
        return func_target

    def is_builtin_name_expr(self, expr: NameExpr) -> bool:
        assert expr.node, "RefExpr not resolved"
        return '.' in expr.node.fullname() and expr.node.fullname().split('.')[0] == 'builtins'

    def load_global(self, expr: NameExpr) -> Value:
        """Loads a Python-level global.

        This takes a NameExpr and uses its name as a key to retrieve the corresponding PyObject *
        from the _globals dictionary in the C-generated code.
        """
        # If the global is from 'builtins', turn it into a module attr load instead
        if self.is_builtin_name_expr(expr):
            return self.load_static_module_attr(expr)
        if self.is_native_module_name_expr(expr) and isinstance(expr.node, TypeInfo):
            assert expr.fullname is not None
            return self.load_native_type_object(expr.fullname)
        _globals = self.add(LoadStatic(object_rprimitive, 'globals', self.module_name))
        reg = self.load_static_unicode(expr.name)
        return self.add(PrimitiveOp([_globals, reg], dict_get_item_op, expr.line))

    def load_static_int(self, value: int) -> Value:
        """Loads a static integer Python 'int' object into a register."""
        static_symbol = self.mapper.literal_static_name(value)
        return self.add(LoadStatic(int_rprimitive, static_symbol, ann=value))

    def load_static_float(self, value: float) -> Value:
        """Loads a static float value into a register."""
        static_symbol = self.mapper.literal_static_name(value)
        return self.add(LoadStatic(float_rprimitive, static_symbol, ann=value))

    def load_static_unicode(self, value: str) -> Value:
        """Loads a static unicode value into a register.

        This is useful for more than just unicode literals; for example, method calls
        also require a PyObject * form for the name of the method.
        """
        static_symbol = self.mapper.literal_static_name(value)
        return self.add(LoadStatic(str_rprimitive, static_symbol, ann=value))

    def load_static_module_attr(self, expr: RefExpr) -> Value:
        assert expr.node, "RefExpr not resolved"
        module = '.'.join(expr.node.fullname().split('.')[:-1])
        name = expr.node.fullname().split('.')[-1]
        left = self.add(LoadStatic(object_rprimitive, 'module', module))
        return self.py_get_attr(left, name, expr.line)

    def load_native_type_object(self, fullname: str) -> Value:
        module, name = fullname.rsplit('.', 1)
        return self.add(LoadStatic(object_rprimitive, name, module, NAMESPACE_TYPE))

    def coerce(self, src: Value, target_type: RType, line: int) -> Value:
        """Generate a coercion/cast from one type to other (only if needed).

        For example, int -> object boxes the source int; int -> int emits nothing;
        object -> int unboxes the object. All conversions preserve object value.

        Returns the register with the converted value (may be same as src).
        """
        if src.type.is_unboxed and not target_type.is_unboxed:
            return self.box(src)
        if ((src.type.is_unboxed and target_type.is_unboxed)
                and not is_same_type(src.type, target_type)):
            # To go from one unboxed type to another, we go through a boxed
            # in-between value, for simplicity.
            tmp = self.box(src)
            return self.unbox_or_cast(tmp, target_type, line)
        if ((not src.type.is_unboxed and target_type.is_unboxed)
                or not is_subtype(src.type, target_type)):
            return self.unbox_or_cast(src, target_type, line)
        return src
