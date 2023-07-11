import math
import torch
import numpy as np

class Value:
    def __init__(self, data, children=(), requieres_grad = True):
        self.requieres_grad = requieres_grad
        self.data = data
        self._prev = set(children)
        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other))
        def _backward():
            if self.requieres_grad:
                self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    #def __sub__(self, other):
    #    other = other if isinstance(other, Value) else Value(other)
    #    out = Value(self.data - other.data, children=(self, other))
    #    def _backward():
    #        self.grad += 1 * out.grad
    #        other.grad += 1 * out.grad
    #    out._backward = _backward
    #
    #    return out

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other))
        def _backward():
            if self.requieres_grad:
                self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, children=(self, other))
        def _backward():
            if self.requieres_grad:
                self.grad += 1 / other.data * out.grad
            other.grad += -self.data / other.data**2 * out.grad
        out._backward = _backward

        return out

    #def __pow__(self, other):
    #    other = other if isinstance(other, Value) else Value(other)
    #    out = Value(self.data ** other.data, children=(self, other))
    #    def _backward():
    #        self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
    #        other.grad += (self.data ** other.data) * math.log(self.data) * out.grad
    #    out._backward = _backward
    #
    #    return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,))

        def _backward():
            if self.requieres_grad:
                self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        #out = Value(0 if self.data < 0 else self.data, (self,))
        out = Value(0 if self.data < 0 else 1, (self,))

        def _backward():
            if self.requieres_grad:
                self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        out = Value(1 / (1 + np.exp(-self.data)), (self,))

        def _backward():
            if self.requieres_grad:
                s = 1 / (1 + math.exp(-self.data))
                self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        _value = (np.exp(self.data) - np.exp(-self.data)) - (np.exp(self.data) + np.exp(-self.data))
        out = Value(_value)

        def _backward():
            if self.requieres_grad:
                self.grad += 1 - _value**2

        out._backward = _backward

        return out

    def MSE(target, output):
        loss = (target - output) ** 2
        loss.backward()
        return loss

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other
    
    def __rmul__(self, other):
        return self * other

    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

#a = Value(2.0)
#b = Value(3.0)
#
#c = -a - b
#
#c.backward()
#
#print(a.grad)
#
#_a = torch.Tensor([2.0]).double(); _a.requires_grad = True
#_b = torch.Tensor([3.0]).double(); _b.requires_grad = True
#
#_c = -_a + _b
#
#_c.backward()
#
#print(_a.grad)