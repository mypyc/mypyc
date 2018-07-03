from mypy.traverser import TraverserVisitor

class FreeVarsVisitor(TraverserVisitor):

	def visit_var(self, o: 'mypy.nodes.Var'):
		pass