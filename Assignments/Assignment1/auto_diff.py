from typing import Any, Dict, List

import numpy as np


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[np.ndarray]
            The input values of the given node.

        Returns
        -------
        output: np.ndarray
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [ mul(output_grad, node.inputs[1]), mul(output_grad, node.inputs[0]) ]         


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise multiplication of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [mul_by_const(output_grad, node.constant)]


class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of input values."""
        """TODO: Your code here"""
        assert len(input_values) == 2 and np.all(input_values[1] != 0) 
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: Your code here"""
        return [div(output_grad, node.inputs[1]), mul(output_grad, div(mul_by_const(node.inputs[0], -1), mul(node.inputs[1], node.inputs[1]))) ]


class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the element-wise division of the input value and the constant."""
        """TODO: Your code here"""
        assert len(input_values) == 1 and node.constant != 0
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        """TODO: Your code here"""
        return [div_by_const(output_grad, node.constant)]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node, trans_A: bool = False, trans_B: bool = False
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix
        trans_A: bool
            A boolean flag denoting whether to transpose A before multiplication.
        trans_B: bool
            A boolean flag denoting whether to transpose B before multiplication.

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"trans_A": trans_A, "trans_B": trans_B},
            name=f"({node_A.name + ('.T' if trans_A else '')}@{node_B.name + ('.T' if trans_B else '')})",
        )

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return the matrix multiplication result of input values.

        Note
        ----
        For this assignment, you can assume the matmul only works for 2d matrices.
        That being said, the test cases guarantee that input values are
        always 2d numpy.ndarray.
        """
        """TODO: Your code here"""
        assert len(input_values) == 2, f"input_values length: %s".format(len(input_values))
        return np.matmul(input_values[0].T if node.attrs["trans_A"] else input_values[0], input_values[1].T if node.attrs["trans_B"] else input_values[1])


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        Note
        ----
        - Same as the `compute` method, you can assume that the input are 2d matrices.
        However, it would be a good exercise to think about how to handle
        more general cases, i.e., when input can be either 1d vectors,
        2d matrices, or multi-dim tensors.
        - You may want to look up some materials for the gradients of matmul.
        """
        """TODO: Your code here"""
        # 根据矩阵乘法的梯度规则和转置标志调整计算方式
        A = node.inputs[0]
        B = node.inputs[1]
        trans_A = node.attrs["trans_A"]
        trans_B = node.attrs["trans_B"]
        
        # 计算A的梯度
        if trans_A:
            grad_A = matmul(B, output_grad, trans_A=trans_B, trans_B=True)
        else:
            grad_A = matmul(output_grad, B, trans_B=not trans_B)
        
        # 计算B的梯度
        if trans_B:
            grad_B = matmul(output_grad, A, trans_A=True, trans_B=trans_A)
        else:
            grad_B = matmul(A, output_grad, trans_A=not trans_A, trans_B=False)
        
        return [grad_A, grad_B]
 
class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.zeros(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]


class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return np.ones(input_values[0].shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """Sum elements along specified axes"""
    
    def __call__(self, node_A: Node, axis: tuple = None) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"axis": axis},
            name=f"Sum({node_A.name}, axis={axis})"
        )
    
    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Sum input values along specified axes"""
        assert len(input_values) == 1
        return np.sum(input_values[0], axis=node.attrs["axis"], keepdims=True)
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Broadcast gradient back to original shape using dynamic shape"""
        # 通过原始输入节点动态获取形状
        return [broadcast_to(output_grad, node.inputs[0], axis=node.attrs.get("axis", None))]

class BroadcastToOp(Op):
    """Broadcast tensor to match target node's shape along specified axes"""
    
    def __call__(self, node_A: Node, target_node: Node, axis: list = None) -> Node:
        return Node(
            inputs=[node_A, target_node],
            op=self,
            attrs={"axis": axis},
            name=f"BroadcastTo({node_A.name}, {target_node.name})"
        )
    
    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Broadcast input to match target node's shape"""
        assert len(input_values) == 2
        input_data = input_values[0]
        target_shape = input_values[1].shape  # 动态获取目标形状
        
        # 直接使用动态获取的目标形状进行广播
        return np.broadcast_to(input_data, target_shape)
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Sum gradients over broadcasted dimensions"""
        # 沿原始广播的轴求和，保持batch维度不变
        return [sum_along_axes(output_grad, axis=node.attrs.get("axis", None))]

class LogarithmOp(Op):
    """Element-wise logarithm operation."""
    
    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"Log({node_A.name})")
    
    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Element-wise logarithm operation"""
        assert len(input_values) == 1
        return np.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Gradient of logarithm is 1/x"""
        return [div(output_grad, node.inputs[0])]

class ExponentiationOp(Op):
    """Element-wise exponentiation operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"Exp({node_A.name})")

    def compute(self, node: Node, input_values: List[np.ndarray]) -> np.ndarray:
        """Element-wise exponentiation operation"""
        assert len(input_values) == 1
        return np.exp(input_values[0])
    
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Gradient of exp(x) is exp(x) * output_grad."""
        return [mul(output_grad, exponentiation(node.inputs[0]))]

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
broadcast_to = BroadcastToOp()
sum_along_axes = SumOp()
logarithm = LogarithmOp()
exponentiation = ExponentiationOp()


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, np.ndarray]) -> List[np.ndarray]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, np.ndarray]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[np.ndarray]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: Your code here"""

        computed_values = {}

        def compute(node):
            if node in computed_values:
                return computed_values[node]
            if node in input_values:
                computed_values[node] = input_values[node]
                return computed_values[node]
            inputs = [compute(inp) for inp in node.inputs]
            computed_values[node] = node.op.compute(node, input_values = inputs)
            return computed_values[node]

        eval_values = []
        for node in self.eval_nodes:
            eval_values.append(compute(node))

        return eval_values


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """

    """TODO: Your code here"""
    visited = set()
    reverse_topo = []
    
    def build_topo(node):
        if node in visited:
            return
        visited.add(node)
        if node.inputs:
            for input_node in node.inputs:
                build_topo(input_node)
        reverse_topo.append(node)
    
    build_topo(output_node)
    
    # Initialize gradients dictionary with the output node's gradient set to 1
    grads = {}
    grads[output_node] = ones_like(output_node)
    
    # Traverse the nodes in reverse topological order to accumulate gradients
    for node in reversed(reverse_topo):
        current_grad = grads.get(node, zeros_like(node))
        if isinstance(node.op, PlaceholderOp):
            continue
        # Compute the gradients for each input of the current node
        input_grads = node.op.gradient(node, current_grad)
        # Update the gradients for each input node
        for input_node, grad in zip(node.inputs, input_grads):
            if input_node in grads:
                grads[input_node] += grad
            else:
                grads[input_node] = grad
    
    # Return the gradients for the requested nodes, defaulting to zero if not found
    return [grads.get(n, zeros_like(n)) for n in nodes]
