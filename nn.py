import matplotlib.pyplot as plt
import random
import numpy as np
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
        self.bias = Value(0)

        self.forward()

    def forward(self, inputs=[]):
        if inputs == []:
            self.activation = sum((wi * xi for wi, xi in zip(self.weights, self.inputs)), self.bias)
        else:
            self.activation = sum((wi * xi for wi, xi in zip(self.weights, inputs)), self.bias)

        if self.activation_function == 'relu':
            self.activation = self.activation.relu()
        elif self.activation_function == 'sigmoid':
            self.activation = self.activation.sigmoid()
        elif self.activation_function == 'tanh':
            self.activation = self.activation.tanh()
            
        return self.activation

    def update(self, lr=0.01):
        for w in self.weights:
            w.data -= w.grad * lr
        
        self.bias.data -= self.bias.grad * lr

    def zero_grad(self):
        for weight in self.weights:
            weight.grad = 0
        
        self.bias.grad = 0

        self.activation.grad = 0
        
    def parameters(self):
        return self.weights + [self.bias]

class Dropout:
    def __init__(self, ratio):
        self.ratio = ratio
        
    def dropout(self, layer):
        length = len(layer.neurons)
        num_elements = round(self.ratio * length)
        indices_to_remove = random.sample(range(length), num_elements)
        modified_layer = [element for index, element in enumerate(layer.neurons) if index not in indices_to_remove]
            
        return modified_layer

class Layer:
    def __init__(self, num_neurons, activation_function, inputs=[]):
        self.activation_function = activation_function
        self.num_neurons = num_neurons
        self.neurons = [Neuron(inputs, activation_function=activation_function) for _ in range(num_neurons)]
        self.activations = []

    def change_inputs(self, inputs=[]):
        for neuron in self.neurons:
            neuron.inputs = inputs

    def forward(self, prev_activations=[]):
        activations = []
        for neuron in self.neurons:
            if prev_activations == []:
                activations.append(neuron.forward())
            else:
                activations.append(neuron.forward(prev_activations))
        self.activations = activations
        
        return self.activations

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
            if isinstance(layer, Dropout):
                print('hi')
                self.layers[-1].neurons = layer.dropout(self.layers[-1])
                continue
                
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
        i = 1
        activations = []
        for layer in self.layers:
            #print(f'forwarding layer {i}...')
            activations = layer.forward(activations)
            i += 1

    # 2 inputs, 1 output :: for now
    def train(self, inputs, targets, epochs, lr=0.01, lr_decay=False, l2reg=False):
        losses = []
        
        if l2reg:
            alpha_l2reg = 1e-4
        
        for e in range(epochs):
            _loss = []
            for x, y in zip(inputs, targets):
                self.inputs = x
                self.layers[0].change_inputs(x)
            
                self.forward()
                output = self.layers[-1].neurons[0].activation
            
                loss = (y[0] - output) ** 2
                _loss.append(loss)
            
            data_loss = sum(_loss)
            if l2reg:
                reg_loss = alpha_l2reg * sum((p*p for p in model.parameters()))
                data_loss += reg_loss
                
            losses.append(data_loss.data)
            data_loss.backward()
            if lr_decay:
                lr = 1.0 - 0.9*e/epochs
            self.update(lr)
            self.zero_grad()
            
            print(f'Epoch {e+1}/{epochs} - loss: {data_loss.data}')

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
        params = []
        for layer in self.layers:
            for neuron in layer.neurons:
                params += neuron.parameters()
                
        return params

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
    [Value(63.0), ], # input layer, define size
    Layer(16, activation_function='sigmoid'), # hidden layer
    #Dropout(0.1),
    Layer(16, activation_function='sigmoid'), # hidden layer
    #Dropout(0.1),
    Layer(1, activation_function='sigmoid'), # output layer
)

#model.train([Value(1.0, requieres_grad=False),], Value(0.1), epochs=100, lr=0.1)
print(len(model.parameters()))
model.train(
    # inputs
    [[Value(1.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(2.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(3.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(4.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(5.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(6.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(7.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(8.0, requieres_grad=False), Value(1.0, requieres_grad=False)],
     [Value(9.0, requieres_grad=False), Value(1.0, requieres_grad=False)],],

    # outputs
    [[Value(0.1, requieres_grad=False)],
     [Value(0.2, requieres_grad=False)],
     [Value(0.3, requieres_grad=False)],
     [Value(0.4, requieres_grad=False)],
     [Value(0.5, requieres_grad=False)],
     [Value(0.6, requieres_grad=False)],
     [Value(0.7, requieres_grad=False)],
     [Value(0.8, requieres_grad=False)],
     [Value(0.8, requieres_grad=False)],],

     epochs=300, 
     #lr=0.3,
     lr_decay=True,
     l2reg=True
)
print('input',model.inputs)
model.forward()
print('output',model.output())
#print(model.parameters())
#model.parameters()
#model.train([Value(1.0), Value(2.0), Value(3.0)], [Value(0.7)], epochs=10, lr=0.1)
#print("output:",model.output(shortcut=False))
#print("parameters")
#model.parameters()


