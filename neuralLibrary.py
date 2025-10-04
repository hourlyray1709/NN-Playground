import numpy as np 
import random 


def LeakyReLU(x, a=0.01): 
    if x > 0: 
        return x 
    else: 
        return a*x

class Neuron: 
    def __init__(self, input_size, weights=None, bias=0, val=0): 
        if weights == None: 
            self.weights = np.array([random.uniform(-1,1) for i in range(input_size)])                  # weights must be same dimension as inputs 
        else: 
            self.weights = np.array(weights)
        self.bias = bias                                                                                # one bias to go into activation func, since we need a1w1 + a2w2 + a3w3 + ... + b 
        self.val = val 
    def forward(self, func, inputs): 
        activationValue = np.dot(inputs, self.weights) + self.bias                                      # a1w1 + a2w2 + a3w3 + ... + b 
        self.val = func(activationValue)                                                                # allow picking custom activation function (e.g. Sigmoid, ReLu)
        return self.val


# testing code 

outputNeuron = Neuron(input_size=2, weights=[2,2], bias=-3)
outputNeuron.forward(LeakyReLU, [1,1])
print(outputNeuron.val)