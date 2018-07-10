from typing import Dict, List, Set

from mypy.nodes import FuncDef, FuncItem, LambdaExpr, NameExpr, SymbolNode, Var
from mypy.traverser import TraverserVisitor


class FreeVariablesVisitor(TraverserVisitor):
    """Class used to visit nested functions and determine free symbols."""
    def __init__(self) -> None:
        super().__init__()
        # Mapping from FuncItem instances to sets of variables. The FuncItem instances are where
        # these variables were first declared, and these variables are free in any functions that
        # are nested within the FuncItem from which they are mapped.
        self.free_variables = {}  # type: Dict[FuncItem, Set[SymbolNode]]
        # Intermediate data structure used to map SymbolNode instances to the FuncDef in which they
        # were first visited.
        self.symbols_to_fitems = {}  # type: Dict[SymbolNode, FuncItem]
        # Stack representing the function call stack.
        self.fitems = []  # type: List[FuncItem]

        self.encapsulating_fitems = set()  # type: Set[FuncItem]
        self.nested_fitems = set()  # type: Set[FuncItem]

    def visit_func_def(self, fdef: FuncDef) -> None:
        # If there were already functions or lambda expressions defined in the function stack, then
        # note the previous FuncItem has containing a nested function and the current FuncDef as
        # being a nested function.
        if self.fitems:
            self.encapsulating_fitems.add(self.fitems[-1])
            self.nested_fitems.add(fdef)

        self.fitems.append(fdef)
        self.visit_func(fdef)
        self.fitems.pop()

    def visit_var(self, var: Var) -> None:
        self.visit_symbol_node(var)

    def visit_symbol_node(self, symbol: SymbolNode) -> None:
        if not self.fitems:
            # If the list of FuncDefs is empty, then we are not inside of a function and hence do
            # not need to do anything regarding free variables.
            return

        if symbol in self.symbols_to_fitems and self.symbols_to_fitems[symbol] != self.fitems[-1]:
            # If the SymbolNode instance has already been visited before, and it was declared in a
            # FuncDef outside of the current FuncDef that is being visted, then it is a free symbol
            # because it is being visited again.
            self.add_free_variable(symbol)
        else:
            # Otherwise, this is the first time the SymbolNode is being visited. We map the
            # SymbolNode to the current FuncDef being visited to note where it was first visited.
            self.symbols_to_fitems[symbol] = self.fitems[-1]

    def visit_lambda_expr(self, expr: LambdaExpr) -> None:
        # If there were already functions or lambda expressions defined in the function stack, then
        # note the previous FuncItem has containing a nested function and the current LambdaExpr as
        # being a nested function.
        if self.fitems:
            self.encapsulating_fitems.add(self.fitems[-1])
            self.nested_fitems.add(expr)

        self.fitems.append(expr)
        self.visit_func(expr)
        self.fitems.pop()

    def visit_name_expr(self, expr: NameExpr) -> None:
        if isinstance(expr.node, (Var, FuncDef)):
            self.visit_symbol_node(expr.node)

    def add_free_variable(self, symbol: SymbolNode) -> None:
        # Get the FuncItem instance where the free symbol was first declared, and map that FuncItem
        # to the SymbolNode representing the free symbol.
        fitem = self.symbols_to_fitems[symbol]
        self.free_variables.setdefault(fitem, set()).add(symbol)
