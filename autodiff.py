import numpy


class Tensor:
    # 让Tensor又操作支持numpy数组，而不是单个数字，参照https://stackoverflow.com/questions/58408999/how-to-call-rsub-properly-with-a-numpy-array
    __array_ufunc__ = None

    def __init__(self, data, from_tensors=None, op=None, grad=None):
        self.data = data  # 数据
        self.from_tensors = from_tensors  # 是从什么Tensor得到的，保存计算图的历史
        self.op = op  # 操作符运算
        # 梯度
        if grad:
            self.grad = grad
        else:
            self.grad = numpy.zeros(self.data.shape) if isinstance(self.data, numpy.ndarray) else 0

    def __add__(self, other):
        # 先判断other是否是常数，然后再调用
        return add.forward([self, other]) if isinstance(other, Tensor) else add_with_const.forward([self, other])

    def __sub__(self, other):
        # other如果是常数，直接调用加法的常数计算
        return sub.forward([self, other]) if isinstance(other, Tensor) else add_with_const.forward([self, -other])

    def __rsub__(self, other):
        # 常数-tensor ，则调用 rsub_with_const
        return rsub_with_const.forward([self, other])

    def __mul__(self, other):
        # 先判断other是否是常数，然后再调用
        return mul.forward([self, other]) if isinstance(other, Tensor) else mul_with_const.forward([self, other])

    def __truediv__(self, other):
        # tensor/常数 则直接使用乘法
        return div.forward([self, other]) if isinstance(other, Tensor) else mul_with_const.forward([self, 1 / other])

    def __rtruediv__(self, other):
        # 常数/tensor，则调用 rdiv_with_const
        return rdiv_with_const.forward([self, other])

    def __neg__(self):
        # 求负直接使用 0-tensor ，即__rsub__
        return self.__rsub__(0)

    def matmul(self, other):
        # 不支持shape为1 的numpy，因为在转置时[1,2]的转置结果仍然为[1,2]，此时需要换成[[1,2]]
        return mul_with_matrix.forward([self, other])

    def sum(self):
        return sum.forward([self])

    def mean(self):
        # 平均使用 求和/数据的量
        return sum.forward([self]) / self.data.size

    def log(self):
        return log.forward([self])

    def exp(self):
        return exp.forward([self])

    def backward(self, grad=None):
        # 判断y的梯度是否存在，如果不存在初始化和y.data一样类型的1的数据
        if grad is None:
            self.grad = grad = numpy.ones(self.data.shape) if isinstance(self.data, numpy.ndarray) else 1
        # 如果op不存在，则说明该Tensor为根节点，其from_tensors也必然不存在，否则计算梯度
        if self.op:
            grad = self.op.backward(self.from_tensors, grad)
        if self.from_tensors:
            for i in range(len(grad)):
                tensor = self.from_tensors[i]
                # 把梯度加给对应的子Tensor，因为该Tensor可能参与多个运算
                tensor.grad += grad[i]
                # 子Tensor进行后向过程
                tensor.backward(grad[i])

    # 清空梯度，训练的时候，每个batch应该清空梯度
    def zero_gard(self):
        self.grad = numpy.zeros(self.data.shape) if isinstance(self.data, numpy.ndarray) else 0

    __radd__ = __add__
    __rmul__ = __mul__


class OP:
    def forward(self, from_tensors):
        pass

    def backward(self, from_tensors, grad):
        pass


class Add(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data + from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad, grad]


class AddWithConst(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data + from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad]


class Sub(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data - from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad, -grad]


class RSubWithConst(OP):
    def forward(self, from_tensors):
        return Tensor(-(from_tensors[0].data - from_tensors[1]), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [-grad]


class Mul(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data * from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad, from_tensors[0].data * grad]


class MulWithConst(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data * from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1] * grad]


class MulWithMatrix(OP):
    def forward(self, from_tensors):
        return Tensor(numpy.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        # Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        return [numpy.matmul(grad, from_tensors[1].data.T), numpy.matmul(from_tensors[0].data.T, grad)]


class Div(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data / from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[1].data,
                -grad * from_tensors[0].data / (from_tensors[1].data * from_tensors[1].data)]


class RDivWithConst(OP):
    def forward(self, from_tensors):
        return Tensor(from_tensors[1] / from_tensors[0].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [-grad * from_tensors[1] / (from_tensors[0].data * from_tensors[0].data)]


class Sum(OP):
    def forward(self, from_tensors):
        return Tensor(numpy.sum(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * numpy.ones(from_tensors[0].data.shape)]


class Exp(OP):
    def forward(self, from_tensors):
        return Tensor(numpy.exp(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * numpy.exp(from_tensors[0].data)]


class Log(OP):
    def forward(self, from_tensors):
        return Tensor(numpy.log(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[0].data]


add = Add()
add_with_const = AddWithConst()
sub = Sub()
rsub_with_const = RSubWithConst()
mul = Mul()
mul_with_const = MulWithConst()
mul_with_matrix = MulWithMatrix()
div = Div()
rdiv_with_const = RDivWithConst()
sum = Sum()
exp = Exp()
log = Log()
