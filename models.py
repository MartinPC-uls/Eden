import eden
import eden.NeuralNetwork as nn

#eden.manual_seed(31234)
eden.set_threads(1)

class Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.dataset = zip(inputs, targets)

class MLP:
    def __init__(self, parameters, *layers):
        self.parameters = parameters
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
    def train(self, epochs, lr, dataset: Dataset):
        for e in range(epochs):
            total_loss = eden.zeros(1,1)
            for input, target in zip(dataset.inputs, dataset.targets):
                predicted = self.forward(input)
                loss = (target - predicted)**2
                total_loss += loss
                #print(loss)
            
            total_loss = eden.Matrix(1.0) / eden.Matrix(len(dataset.inputs)) * total_loss
            self.parameters.zero_grad()
            total_loss.backward()
            self.parameters.update(lr)
            
            if e % 100 == 0:
                print(f"Epoch {(e+1)}/{epochs} completed - loss: {total_loss}")

model_parameters = nn.Parameters()
mlp_model = MLP(model_parameters, 
    nn.Linear(3, 3, model_parameters, activation_fn=eden.Sigmoid),
    nn.Linear(3, 3, model_parameters, activation_fn=eden.relu),
    nn.Linear(3, 3, model_parameters, activation_fn=eden.Softsign),
    nn.Linear(3, 1, model_parameters, activation_fn=eden.Sigmoid),
)

inputs = [eden.Matrix([1.0, 2.0, 3.0]),
          eden.Matrix([4.0, 5.0, 6.0]),
          eden.Matrix([7.0, 8.0, 9.0]),
          eden.Matrix([10.0, 11.0, 12.0]),
          eden.Matrix([13.0, 14.0, 15.0]),
          eden.Matrix([16.0, 17.0, 18.0]),
          eden.Matrix([19.0, 20.0, 21.0])]

outputs = [eden.Matrix(0.06),
           eden.Matrix(0.15),
           eden.Matrix(0.24),
           eden.Matrix(0.33),
           eden.Matrix(0.42),
           eden.Matrix(0.51),
           eden.Matrix(0.60)]

dataset = Dataset(inputs, outputs)

import time

output1 = mlp_model.forward([1.0, 2.0, 3.0])
output2 = mlp_model.forward([13.0, 14.0, 15.0])
output3 = mlp_model.forward([19.0, 20.0, 21.0])
print(output1) # 0.06
print(output2) # 0.42
print(output3) # 0.6

start = time.time()
mlp_model.train(3000, 0.1, dataset)
end = time.time()
total_time = end-start

print(f"Training took {total_time} seconds.")

output1 = mlp_model.forward([1.0, 2.0, 3.0])
output2 = mlp_model.forward([4.0, 5.0, 6.0])
output3 = mlp_model.forward([13.0, 14.0, 15.0])
output4 = mlp_model.forward([19.0, 20.0, 21.0])
print(output1) # 0.06
print(output2) # 0.42
print(output3) #
print(output4) # 0.6

