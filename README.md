# Eden
Autograd engine based on Andrej Karpathy's Micrograd and PyTorch

Github page: https://github.com/karpathy/micrograd

# Goal of this project
I just wanted to make this library for personal educational purposes and for fun, that is, knowing how a Neural Network works under the hood. I will not recreate PyTorch or TensorFlow, I believe they're great APIs for Deep Learning and should not be replaced.

# What I've learned so far...
* Some SIMD operations with AVX
* Usage of Clang flags for compiler optimizations
* Optimizations for matrix operatations using multithreading
* DAG (Directed Acyclic Graph) applied to Neural Networks, making a Computational Graph, keeping track of every operation done with the matrices
* C++
* Python and ctypes to link C code

# How it works
Eden is very similar to PyTorch, in the sense that it uses the same system for Autograd: A Computational Graph.

![computational_graph](https://github.com/MartinPC-uls/Eden/assets/64845110/9c72e019-29fa-4950-b753-f44c6f3637e6)

(Image source: https://www.geeksforgeeks.org/computational-graphs-in-deep-learning/)

This graph keeps track of every operation that's done with the class Matrix in Python. This approach makes it really simple to derive a neural network and to scale them. Backpropagation is based on the chain rule of calculus, which is the base of the autograd system.

# Examples
Import the package:
~~~python
import eden
~~~

Create matrices and do some operations with them.
~~~python
a = eden.Matrix([[1, 2, 3],
                 [4, 5, 6]])

b = eden.Matrix([[7, 8, 9],
                 [10, 11, 12]])

# Element-wise multiplication
c = a * b
# output: Matrix([[7., 16., 27.],
#                 [40., 55., 72.]])


# Matrix multiplication, to do so,
# we need to transpose a matrix, in this case, the first one. (Columns of 'a' must be equal to rows of 'b')
c = a.T @ b
# output: Matrix([[47., 52., 57.],
#                 [64., 71., 78.],
#                 [81., 90., 99.]])
~~~

You can also create Scalar values or Vector with Matrix class, although, under the hood everything is a Matrix.
~~~python
a = eden.Matrix(1) # Scalar value of 1
b = eden.Matrix([1, 2, 3]) # Vector of values [1, 2, 3]

# Check shapes
a.shape # (1, 1)
b.shape # (1, 3)
~~~

You might also want to create a matrix of random values:
~~~python
a = eden.randn(100, 200) # Matrix of shape (100, 200) with random values based on Standard Normal Distribution.
b = eden.random(50, 62) # Matrix of shape (50, 62) with random values (no distribution).
~~~

# Autograd examples
Autograd is the base system that will bring neural networks to life! Let's look at some simple examples:
~~~python
eden.manual_seed(56413) # Set a seed for random number generator

a = eden.randn(10, 22, requires_grad=True)
b = eden.randn(22, 10, requires_grad=True)
c = a @ b
d = eden.sum(c) # Single value, sums all the elements in the matrix
~~~

And now, you need to derive by hand...

Of course you don't, we've got Autograd, and to "backpropagate", as you do in neural networks for training and calculate all the gradients for weights and biases, you can do the following:
~~~python
d.backward()
~~~
And with that, you will get all the respectives gradients for every single node of the graph.
You can ONLY do "backward()" on Scalar values (shape 1x1) for now, and also one of the children will need to have "requires_grad" on True. You can specify the gradient requirement on creation time.

# Neural Networks
Eden has its own custom module, similar to PyTorch, to start creating Neural Networks.

~~~python
import eden.NeuralNetwork as nn
~~~

Let's create a simple MLP model. First, we'll start by creating a simple dataset with the Dataset class.
~~~python
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

dataset = nn.Dataset(inputs, outputs)
~~~

Now, let's initialize a Paramater class that will contain all the parameters of the MLP model.
~~~python
model_parameters = nn.Parameters()
~~~

Now it is time to attack, we'll create the MLP model.
~~~python
mlp_model = nn.MLP(model_parameters, 
    nn.Linear(3, 3, model_parameters, activation_fn=eden.Sigmoid),
    nn.Linear(3, 3, model_parameters, activation_fn=eden.ReLU),
    nn.Linear(3, 3, model_parameters, activation_fn=eden.Softsign), # Eden also supports Softsign activation function, proposed in Xavier's paper.
    nn.Linear(3, 1, model_parameters, activation_fn=eden.Sigmoid),
)
~~~
Xavier's Paper: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf ("Understanding the difficulty of training deep feedforward neural networks")

'nn.MLP' takes as argument the parameters and the Linear layers.
'nn.Linear' is a linear transformation of the incoming data. Takes as arguments number of inputs (first, 3 in this case), and number of outputs (seconds, 3 in this case).
So this model has an Input Layer with 3 inputs, 3 Hidden Layers with 3 neurons each, and an Output Layer of 1 neuron. All of them takes the model_parameters as argument, to store its weights and biases. We need to specify an activation function, if we don't do it, it'll assign an activation function called "TLU", which is basically the value that you get out of weights, biases and inputs.
The activations functions supported at the moment are:
* ReLU (bounds: 0, inf)
* Sigmoid (bounds: 0, 1)
* Tanh (bounds: -1, 1)
* Softsign (bounds: -0.8, 0.8)

Now, the final step: train the neural network.
~~~python
mlp_model.train(epochs=3000, lr=0.1, dataset)
~~~

And there you go! and to get prediction out of that, you can simply call the forward function:
~~~python
prediction = mlp_model.forward([1.0, 2.0, 3.0]) # input without calling Matrix class
~~~

Take in mind that this project isn't the best and it's not really serious, but you're free to do some modifications to the code as needed.

I will keep updating the library, but not too much! I need to move forward with other serious projects.
