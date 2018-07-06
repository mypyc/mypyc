from typing import Dict, List, Set

from mypy.nodes import FuncDef, FuncItem, LambdaExpr, NameExpr, SymbolNode, Var
from mypy.traverser import TraverserVisitor


class FreeSymbolsVisitor(TraverserVisitor):
    """Class used to visit nested functions and determine free symbols."""
    def __init__(self) -> None:
        super().__init__()
        self.free_symbols = {}  # type: Dict[FuncItem, Set[SymbolNode]]
        self.symbols_to_fitems = {}  # type: Dict[SymbolNode, FuncItem]
        self.fitems = []  # type: List[FuncItem]
        self.encapsulating_fitems = set()  # type: Set[FuncItem]
        self.nested_fitems = set()  # type: Set[FuncItem]

    def visit_func_def(self, fdef: FuncDef) -> None:
        # If there were already functions defined in the function stack, then note the previous
        # FuncDef has containing a nested function and the current FuncDef as being a nested
        # function.
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
            self.add_free_symbol(symbol)
        else:
            # Otherwise, this is the first time the SymbolNode is being visited. We map the
            # SymbolNode to the current FuncDef being visited to note where it was first visited.
            self.symbols_to_fitems[symbol] = self.fitems[-1]

    def visit_lambda_expr(self, expr: LambdaExpr) -> None:
        if self.fitems:
            self.encapsulating_fitems.add(self.fitems[-1])
            self.nested_fitems.add(expr)

        self.fitems.append(expr)
        self.visit_func(expr)
        self.fitems.pop()

    def visit_name_expr(self, expr: NameExpr) -> None:
        if isinstance(expr.node, (Var, FuncDef)):
            self.visit_symbol_node(expr.node)

    def add_free_symbol(self, symbol: SymbolNode) -> None:
        # Get the FuncDef instance where the free symbol was first declared, and map that FuncDef
        # to the SymbolNode representing the free symbol.
        fdef = self.symbols_to_fitems[symbol]
        if fdef not in self.free_symbols:
            self.free_symbols[fdef] = set()
        self.free_symbols[fdef].add(symbol)
