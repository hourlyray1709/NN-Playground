import numpy as np 
import random 


def LeakyReLU(x, a=0.01): 
    return np.where(x > 0, x, a * x)

def MSE(predicted, actual): 
    actual = np.array(actual)
    error = np.mean((predicted - actual)**2)
    return error 

class Layer: 
    def __init__(self, num_neurons, input_size, activation_func, override_weights=None, override_bias=None, loss_func=None): 
        self.activation_func = activation_func
        self.loss_func = None 

        if override_weights == None: 
            self.weights = np.random.uniform(-1, 1, (input_size, num_neurons,))                              # weights[4, 0] means the weight going from 4th input to 0th output neuron
        else: 
            self.weights = np.array(override_weights)

        if override_bias == None: 
            self.biases = np.zeros(num_neurons,)
        else: 
            self.biases = np.array(override_bias)

        self.output = np.zeros(num_neurons,)

        self.weight_gradients = np.zeros((input_size, num_neurons))
        self.bias_gradients = np.zeros(num_neurons)
    def forward(self, inputs): 
        inputs = np.array(inputs)
        z = np.dot(inputs, self.weights) + self.biases 
        self.output = self.activation_func(z)
        return self.output
    def backward(self, outputs): 
        MSE_error = MSE(self.output, outputs)
        


# testing code 

ANDLayer = Layer(num_neurons=1, input_size=2, activation_func=LeakyReLU)
ANDLayer.forward([1,1])
print(ANDLayer.output)