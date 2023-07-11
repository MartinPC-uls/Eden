import matplotlib.pyplot as plt
import random
import numpy as np
#from autograd.engine import Value
import sys
sys.path.append('/Users/ghanvert/Eden')

from autograd.engine import Value

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, inputs, activation_function):
        self.activation_function = activation_function
        self.inputs = [] # list of Value objects at the end.
        if isinstance(inputs, list) and all(isinstance(item, Neuron) for item in inputs):
            self.inputs = [neuron.activation for neuron in inputs]
        elif isinstance(inputs, list) and all(isinstance(item, Value) for item in inputs):
            self.inputs = inputs
        else:
            raise Exception("Eden only supports Value and Neuron object types for 'inputs'")

        self.weights = [Value(random.uniform(-1, 1)) for _ in range(len(inputs))]
        #self.weights = [Value(0) for _ in range(len(inputs))]
        self.bias = Value(random.uniform(-1, 1))

        self.forward()

    def forward(self):
        self.activation = sum((wi * xi for wi, xi in zip(self.weights, self.inputs)), self.bias)
        if self.activation_function == 'relu':
            self.activation = self.activation.relu()
        elif self.activation_function == 'sigmoid':
            self.activation = self.activation.sigmoid()
        elif self.activation_function == 'tanh':
            self.activation = self.activation.tanh()

    def update(self, lr=0.01):
        for w in self.weights:
            w.data -= w.grad * lr
        
        self.bias.data -= self.bias.grad * lr

    def zero_grad(self):
        for weight in self.weights:
            weight.grad = 0
        
        self.bias.grad = 0

        self.activation.grad = 0

class Layer:
    def __init__(self, num_neurons, activation_function, inputs=[]):
        self.activation_function = activation_function
        self.num_neurons = num_neurons
        self.neurons = [Neuron(inputs, activation_function=activation_function) for _ in range(num_neurons)]

    def change_inputs(self, inputs=[]):
        for neuron in self.neurons:
            neuron.inputs = inputs

    def forward(self):
        for neuron in self.neurons:
            neuron.forward()

    def update(self, lr=0.01):
        for neuron in self.neurons:
            neuron.update(lr)

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

class MLP:
    def __init__(self, inputs, *args):
        self.inputs = inputs
        self.layers = []
        self.args = args
        aux = inputs
        for layer in args:
            _layer = Layer(layer.num_neurons, inputs=aux, activation_function=layer.activation_function)
            self.layers.append(_layer)
            aux = _layer.neurons

    def restart(self):
        self.layers = []
        aux = self.inputs
        for layer in self.args:
            _layer = Layer(layer.num_neurons, inputs=aux, activation_function=layer.activation_function)
            self.layers.append(_layer)
            aux = _layer.neurons

    def update(self, lr=0.01):
        for layer in self.layers:
            layer.update(lr)

    def forward(self):
        for layer in self.layers:
            layer.forward()

    # 2 inputs, 1 output :: for now
    def train(self, inputs, target, epochs, lr=0.01):
        losses = []

        for input, output in zip(inputs, target):
            x = Value(input.data)
            y = Value(output.data)
        
            self.inputs = input
            self.layers[0].change_inputs(x)
        
            self.forward()
            output = self.layers[-1].neurons[0].activation
        
            self.zero_grad()
        
            loss = (y - x) ** 2
            loss.backward()
        
            self.update(lr)
        
            losses.append(loss.data)


        #for _ in range(epochs):
        #    self.forward()
        #    output = self.layers[-1].neurons[0].activation
        #    
        #    self.zero_grad()
        #
        #    loss = (target - output) ** 2
        #    loss.backward()
        #    
        #    self.update(lr)
        #    
        #    losses.append(loss.data)

        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()


    def zero_grad(self):
        for input in self.inputs:
            input.grad = 0

        for layer in self.layers:
            layer.zero_grad()

    def parameters(self):
        for layer in self.layers:
            print("-- Layer --")
            for neuron in layer.neurons:
                print('weights',neuron.weights)
                print('bias',neuron.bias)
                print('== Neuron activation:', neuron.activation)

    def output(self, shortcut=False):
        if shortcut == False:
            self.forward()
        else:
            self.forward_cut()
        return self.layers[-1].neurons[0].activation.data

def get_middle_value(sorted_array):
    length = len(sorted_array)
    middle_index = length // 2

    if length % 2 == 0:
        # If the length is even, return the average of the two middle values
        return (sorted_array[middle_index - 1] + sorted_array[middle_index]) / 2
    else:
        # If the length is odd, return the middle value
        return sorted_array[middle_index]


model = MLP(
    [Value(1.0), ], # input layer, define size
    Layer(5, activation_function='sigmoid'), # hidden layer
    #Layer(5, activation_function='sigmoid'),
    Layer(1, activation_function='sigmoid'), # output layer
)

#model.train([Value(1.0, requieres_grad=False),], Value(0.1), epochs=100, lr=0.1)
model.train(
    # inputs
    [Value(1.0, requieres_grad=False),
     Value(2.0, requieres_grad=False),
     Value(3.0, requieres_grad=False),
     Value(4.0, requieres_grad=False),
     Value(5.0, requieres_grad=False),
     Value(6.0, requieres_grad=False),
     Value(7.0, requieres_grad=False),
     Value(8.0, requieres_grad=False),
     Value(9.0, requieres_grad=False),
     Value(10.0, requieres_grad=False),
     Value(11.0, requieres_grad=False),],

    # outputs
    [Value(1.0),
     Value(0.0),
     Value(1.0),
     Value(0.0),
     Value(1.0),
     Value(0.0),
     Value(1.0),
     Value(0.0),
     Value(1.0),
     Value(0.0),
     Value(1.0),
     Value(0.0),],

     epochs=100, lr=0.1
)
#model.train([Value(1.0), Value(2.0), Value(3.0)], [Value(0.7)], epochs=2000, lr=0.1)
print("output:",model.output(shortcut=False))
print("parameters")
model.parameters()


