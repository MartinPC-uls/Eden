import eden

class Linear:
    def __init__(self, n_inputs, n_outputs, parameters, requires_bias=True, activation_fn=eden.TLU, requires_grad=True):
        self.weights = eden.randn(n_outputs, n_inputs, requires_grad=requires_grad)
        self.requires_bias = requires_bias
        if requires_bias:
            self.biases = eden.randn(1, n_outputs, requires_grad=requires_grad)
        
        parameters.append(self.weights, self.biases)
        self.activation_fn = activation_fn
        
    def __call__(self, input) -> eden.Matrix:
        if self.requires_bias:
            return self.activation_fn(self.weights @ input + self.biases)
        
        return self.activation_fn(self.weights @ input)