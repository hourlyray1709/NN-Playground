import numpy as np 
import random 


def LeakyReLU(x, a=0.01): 
    return np.where(x > 0, x, a * x)

class Layer: 
    def __init__(self, num_neurons, input_size, activation_func, override_weights=None, override_bias=None): 
        self.activation_func = activation_func
        if override_weights == None: 
            self.weights = np.random.uniform(-1, 1, (input_size, num_neurons,))                              # weights[4, 0] means the weight going from 4th input to 0th output neuron
        else: 
            self.weights = np.array(override_weights)
        if override_bias == None: 
            self.biases = np.zeros(num_neurons,)
        else: 
            self.biases = np.array(override_bias)
        self.output = np.zeros(num_neurons,)
    def forward(self, inputs): 
        inputs = np.array(inputs)
        z = np.dot(inputs, self.weights) + self.biases 
        self.output = self.activation_func(z)
        return self.output

# testing code 

ANDLayer = Layer(num_neurons=1, input_size=2, activation_func=LeakyReLU, override_weights=[[2],[2]], override_bias=[-3])
ANDLayer.forward([1,1])
print(ANDLayer.output)