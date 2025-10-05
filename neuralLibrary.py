import numpy as np 
import random 
import matplotlib.pyplot as plt 

def LeakyReLU(x, a=0.01): 
    return np.where(x > 0, x, a * x)

def LeakyReLU_grad(x, a=0.01): 
    return np.where(x>0, 1, a)

def tanh(x): 
    return np.tanh(x)

def tanhGrad(x): 
    return 1/(np.cosh(x)**2)

def MSE(predicted, actual): 
    actual = np.array(actual)
    error = np.mean((predicted - actual)**2)
    return error 

def MSE_grad(predicted, actual): 
    actual = np.array(actual)
    return 2 * (predicted - actual)

activationFuncToGradFunc = {
    LeakyReLU : LeakyReLU_grad,
    tanh: tanhGrad,
}

class Layer: 
    def __init__(self, num_neurons, input_size, activation_func, override_weights=None, override_bias=None, loss_func=None): 
        self.input_size = input_size 
        self.num_neurons = num_neurons

        self.activation_func = activation_func
        self.loss_func = None 

        if override_weights == None: 
            self.weights = np.random.uniform(-1, 1, (input_size, num_neurons,))                              # weights[4, 0] means the weight going from 4th input to 0th output neuron
        else: 
            self.weights = np.array(override_weights)

        if override_bias == None: 
            self.biases = np.random.uniform(-0.1, 0.1, (num_neurons))
        else: 
            self.biases = np.array(override_bias)

        self.output = np.zeros(num_neurons,)

        self.batch_weight_gradients = np.zeros((input_size, num_neurons))
        self.batch_bias_gradients = np.zeros(num_neurons)
        self.trainEx_weight_gradients = np.zeros((input_size, num_neurons))
        self.trainEx_bias_gradients = np.zeros(num_neurons)
        self.batchSize = 0 
    def forward(self, inputs): 
        inputs = np.array(inputs)
        z = np.dot(inputs, self.weights) + self.biases
        self.z = z                                             # store feedforward z values for backprop 
        self.inputs = inputs                                          
        self.output = self.activation_func(z)
        return self.output
    def outlayer_backward(self, outputs): 
        MSE_error = MSE(self.output, outputs)
        MSE_gradient = MSE_grad(self.output, outputs)         # MSE grad is the deriv of Loss wrt activation, shape (num_neurons,)
        actDerivFunc = activationFuncToGradFunc[self.activation_func]  # use the correct gradient function depending on actFuc
        actFuncGradient = actDerivFunc(self.z)           # deriv of activation wrt to feedforward z value, shape (num_neurons,)
        delta = MSE_gradient * actFuncGradient           # derive of loss wrt to z value, shape (num_neurons,)
        self.delta = delta               
        
        self.trainEx_weight_gradients = np.outer(self.inputs, delta)     # stores gradients for single training Example - trainEx
        self.trainEx_bias_gradients = delta 
        self.loss = MSE_error                                            # save loss for loss visualisation later 
    def hidlayer_backward(self, nextLayer): 
        actDerivFunc = activationFuncToGradFunc[self.activation_func]    # calculate deriv of act function wrt to inputs 
        actFuncGradient = actDerivFunc(self.z).flatten()
        next_delta = nextLayer.delta.flatten()                      
        delta = np.dot(nextLayer.weights, next_delta) * actFuncGradient  # find deriv of cost wrt to own activation 
        self.delta = delta 

        self.trainEx_weight_gradients = np.outer(self.inputs, delta)
        self.trainEx_bias_gradients = delta 
    def addGradients(self): 
        self.batch_weight_gradients += self.trainEx_weight_gradients      # sum up gradients for entire batch 
        self.batch_bias_gradients += self.trainEx_bias_gradients.flatten()
        self.batchSize += 1                                               # used to calculate average gradient over batch 
    def resetGradients(self): 
        input_size = self.input_size 
        num_neurons = self.num_neurons
        self.batch_weight_gradients = np.zeros((input_size, num_neurons))
        self.batch_bias_gradients = np.zeros(num_neurons)
        self.trainEx_weight_gradients = np.zeros((input_size, num_neurons))
        self.trainEx_bias_gradients = np.zeros(num_neurons)
        self.batchSize = 0 
    def applyGradients(self, learnRate=0.01):                                        # adjust weights according gradients in batch 
        self.weights -= learnRate * self.batch_weight_gradients / self.batchSize
        self.biases -= learnRate * self.batch_bias_gradients / self.batchSize 
        self.resetGradients()


class Network: 
    def __init__(self, shape, input_size, activation_funcs=[]): 

        if activation_funcs == []: 
            self.layers = [Layer(shape[0], input_size, LeakyReLU)]
        else: 
            self.layers = [Layer(shape[0], input_size, activation_funcs[0])]
        for i in range(1, len(shape)):
            if activation_funcs == []: 
                self.layers.append(Layer(shape[i], shape[i-1], LeakyReLU))
            else: 
                self.layers.append(Layer(shape[i], shape[i-1], activation_funcs[i]))
        
    def forward(self, inputs): 
        for i in self.layers: 
            inputs = i.forward(inputs)
        self.output = inputs
        return inputs
    
    def backward(self, output): 
        self.layers[-1].outlayer_backward(output)

        for i in range(len(self.layers)-2, -1, -1): 
            self.layers[i].hidlayer_backward(self.layers[i+1])
        
        for i in self.layers: 
            i.addGradients()
    
    def applyGradients(self, learnRate=0.05): 
        for i in self.layers: 
            i.applyGradients(learnRate)
    
    def train(self, dataInputs, dataOutputs, batchSize, epochs, learnRate=0.05, visualise=False): 
        if len(dataInputs) != len(dataOutputs): 
            print("WARNING: Length of training data does not match labels! Cannot train neural network")
            return None 

        n = len(dataInputs)
        batches = n // batchSize
        losses = [] 

        data = {dataInputs[i] : dataOutputs[i] for i in range(n)}
        random.shuffle(dataInputs)
        for epoch in range(epochs): 
            loss = 0 
            for i in range(0, len(dataInputs), batchSize): 
                for j in range(i, i+batchSize): 
                    self.forward(dataInputs[j])
                    self.backward(data[dataInputs[j]])
                    loss += self.layers[-1].loss 
                self.applyGradients(learnRate)
            if visualise: 
                losses.append(loss / len(dataInputs))
        
            print("Epoch: {}, Loss: {}".format(epoch, loss / len(dataInputs)))
        if visualise: 
            x_axis = np.arange(epochs)
            plt.plot(x_axis, losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()




def XORExample():
    XORNet = Network([8,8,1], 2)

    dataInputs = [(0,0), (0,1), (1,0), (1,1)] 
    dataOutputs = [0, 1, 1, 0] 

    XORNet.train(dataInputs, dataOutputs, 4, 1000, visualise=True)
    XORNet.forward([1,1])
    print(XORNet.output)

def NANDExample(): 
    NANDNet = Network([4,4,1], 2)

    dataInputs = [(0,0), (0,1), (1,0), (1,1)] 
    dataOutputs = [1, 1, 1, 0]

    NANDNet.train(dataInputs, dataOutputs, 4, 1000, visualise=True)
    NANDNet.forward([1,1])
    print(NANDNet.output)

def sinExample(): 
    sinNet = Network([16,16,8,1], 1, activation_funcs=[tanh, tanh, tanh, tanh])

    dataInputs = np.linspace(-2*np.pi, 2*np.pi, 1000)
    dataOutputs = [np.sin(i) for i in dataInputs]

    sinNet.train(dataInputs, dataOutputs, 50, 400, visualise=False)
    sinNet.forward([0])
    print(sinNet.output)

    # Generate inputs
    x_vals = np.linspace(-2*np.pi, 2*np.pi, 500)
    y_true = np.sin(x_vals)

    # Get network predictions
    y_pred = [sinNet.forward([x])[0] for x in x_vals]

    # Plot
    plt.plot(x_vals, y_true, label="True sin(x)")
    plt.plot(x_vals, y_pred, label="NN approximation", linestyle='dashed')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Neural Network vs True sin(x)")
    plt.legend()
    plt.show()

sinExample()
