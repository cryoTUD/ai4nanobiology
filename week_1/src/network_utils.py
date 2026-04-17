import numpy as np 

def activation_function(z, function_type="sigmoid"):
    if function_type == "sigmoid":
        return 1 / (1 + np.exp(-z))
    elif function_type == "relu":
        return np.maximum(0, z)
    elif function_type == "tanh":
        return np.tanh(z)
    elif function_type == "linear":
        return z
    else:
        raise ValueError("Unsupported activation function.")

class Neuron:
    def __init__(self, weights=None, bias=None, input_vector=None, activation_function_type="relu"):
        self.weights = weights if weights is not None else np.random.normal(0, 1) # initialize weights randomly if not provided
        self.bias = bias if bias is not None else np.random.normal(0, 1)
        self.input_vector = input_vector
        self.activation_function_type = activation_function_type
        self.forward_gradient = None # to store the gradient of the neuron it connects to in the forward pass
        self.backward_gradient = None # to help the gradient of the neuron it connects to in the backward pass

    def forward(self):
        z = np.dot(self.weights, self.input_vector) + self.bias
        output = activation_function(z, function_type=self.activation_function_type)
        output = output.squeeze()  # Remove any extra dimensions
        return output

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0).squeeze()
    
    def backward(self):
        # compute the gradient of the output of this neuron with respect to its weights and bias using the chain rule
        all_weights = np.array(self.weights)
        all_inputs = np.array(self.input_vector)
        
        dy_dw = []
        dyi_dy = []
        for i, x_input in enumerate(all_inputs):
            # reshape weights and inputs to ensure they are 1D arrays for dot product if they are not already
            z = np.dot(all_weights, all_inputs) + self.bias
            if self.activation_function_type == 'relu':
                dy_dz = self.relu_derivative(z)
            elif self.activation_function_type == 'linear':
                dy_dz = 1
            else:
                raise ValueError("Unsupported activation function for backward pass.")
            dz_dw = x_input
            dz_dx = all_weights[i]
            dy_dw.append(dy_dz * dz_dw)
            dyi_dy.append(dy_dz * dz_dx) # backward gradient

        dy_db = self.relu_derivative(z) * 1

        dL_dw = self.forward_gradient * np.array(dy_dw)
        dL_db = self.forward_gradient * dy_db
        dL_dy = self.forward_gradient * np.array(dyi_dy)

        self.gradient_wrt_weights = np.array(dL_dw) 
        self.gradient_wrt_bias = np.array(dL_db)
        self.backward_gradient = np.array(dL_dy)
    
    def update(self, learning_rate=0.01):
        # update weights and bias based on the loss gradient and learning rate 
        self.weights = self.weights - learning_rate * self.gradient_wrt_weights
        self.bias = self.bias - learning_rate * self.gradient_wrt_bias


class Layer:
    def __init__(self, layer_type="hidden"):
        self.neurons = []
        self.forward_layer = None 
        self.backward_layer = None 
        self.outputs = []
    def add_neuron(self, neuron):
        # to add single neuron to the layer
        self.neurons.append(neuron)
    def add_neurons(self, neurons):
        # to add multiple neurons to the layer
        self.neurons.extend(neurons)
    
    def forward(self):
        outputs = []
        for neuron in self.neurons:
            output = neuron.forward()
            outputs.append(output)
        
        outputs = np.array(outputs).T
        #outputs = outputs.squeeze() 
        self.outputs = outputs
        return outputs



#######
def plot_loss_landscape_with_state(loss_fn, output_vector, states=None, window_size=100, tangent=None, \
                                   show_legend=True):
    import matplotlib.pyplot as plt
    y_pred_around_output = np.linspace(output_vector-window_size, output_vector+window_size, 1000)
    loss_values = [loss_fn(output_vector, y_pred) for y_pred in y_pred_around_output]
    plt.figure(figsize=(3,3))
    plt.plot(y_pred_around_output, loss_values)
    plt.xlabel("Predicted values")
    plt.ylabel("Loss")

    # Add points
    if states is not None:
        for state in states.values():
            plt.scatter(state['prediction'], state['loss'], label=state['condition'])

    if tangent is not None:
        # tangent = slope
        tangent_line = tangent['slope'] * y_pred_around_output + (tangent['loss'] - tangent['slope'] * tangent['prediction'])
        plt.plot(y_pred_around_output, tangent_line, color="gray", \
                 linestyle="--", label=r'$\delta_{out}$' + '={:.2f}'.format(tangent['slope']))
    
    plt.ylim(-200, max(loss_values)*1.1)
    if show_legend:
        plt.legend()
        

def print_summary_of_network(**layers):
    import pandas as pd
    df = pd.DataFrame(columns=["Layer", "Neuron", "Weights", "Bias", "Activation Function", "Input Vector", "Output"])
    for layer_name, layer in layers.items():
        df_layer = pd.DataFrame(columns=["Layer", "Neuron", "Weights", "Bias", "Activation Function", "Input Vector", "Output"])
        df_layer["Layer"] = [layer_name] * len(layer.neurons)
        df_layer["Neuron"] = [f"Neuron {i+1}" for i in range(len(layer.neurons))]
        df_layer["Weights"] = [neuron.weights for neuron in layer.neurons]
        df_layer["Bias"] = [neuron.bias for neuron in layer.neurons]
        df_layer["Activation Function"] = [neuron.activation_function_type for neuron in layer.neurons]
        df_layer["Input Vector"] = [neuron.input_vector for neuron in layer.neurons]
        df_layer["Output"] = [neuron.forward() for neuron in layer.neurons]
        df = pd.concat([df, df_layer], ignore_index=True)
    
    # round the weights, bias, input vector and output to 2 decimal places for better display
    df["Weights"] = df["Weights"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Bias"] = df["Bias"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Input Vector"] = df["Input Vector"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Output"] = df["Output"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    
    print(df)

        # print("-" * 20)
        # print(f"Layer: {layer_name}")
        # print("-" * 20)
        # for i, neuron in enumerate(layer.neurons):
        #     print(f" Neuron {i+1}:")
        #     print(f"    Weights: {neuron.weights}", end=" | ")
        #     print(f"    Bias: {neuron.bias}", end=" | ")
        #     print(f"    Activation Function: {neuron.activation_function_type}", end=" | ")
        #     print(f"    Input Vector: {neuron.input_vector}", end=" | ")
        #     print(f"    Output: {neuron.forward()}")
