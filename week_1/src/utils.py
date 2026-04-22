import os 
import numpy as np


def upload_to_surfdrive(file_path, student_id):
    # upload the model to surfdrive 
    import os
    import requests
    from datetime import datetime

    """Upload a file to a SURFdrive file-drop share."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    surfdrive_link = "https://surfdrive.surf.nl/s/cxQ74XXfRXCKkZJ"

    # Extract share token from the link (the last path segment)
    share_token = surfdrive_link.rstrip("/").split("/")[-1]
    
    # Build a unique remote filename so submissions don't collide
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.basename(file_path)
    remote_name = f"{student_id}_{timestamp}_{original_name}"
    
    # WebDAV upload endpoint — filename goes in the URL
    upload_url = f"https://surfdrive.surf.nl/public.php/webdav/{remote_name}"
    
    with open(file_path, 'rb') as f:
        response = requests.put(
            upload_url,
            data=f,                              # raw bytes, NOT files=
            auth=(share_token, ""),  # token as username
        )
    
    if response.status_code in (200, 201, 204):
        print(f"✓ Uploaded as: {remote_name}")
    else:
        print(f"✗ Upload failed (HTTP {response.status_code})")
        print(f"  Response: {response.text[:300]}")
        
## Util functions for module 2
def degree_3_polynomial(x, coeffs, noise_std=0):
    A, B, C, D = coeffs
    return A + B*x + C*x**2 + D*x**3 + np.random.normal(0, noise_std, size=x.shape)

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


def update_neuron_output_using_activation_point(slope, function_type, activation_point, x):
    # Plot the output
    from netgraph import Graph
    import matplotlib.pyplot as plt
    x_global = np.linspace(-10, 10, 400)
    # calculate bias from activation point
    bias = -slope * activation_point
    z = slope * x_global + bias
    y_global = activation_function(z, function_type=function_type)
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 3))
    # set ax to 3,3 
    ax.plot(x_global, y_global, label=f"Output")
    ax.set_xlabel('Input (x)')
    ax.set_ylabel('Output (y)')
    ax.set_xlim(-10, 10)
    #plt.xticks(np.arange(-4, 11, 4))
    if function_type in ["relu"]:
        y_min = -2.5
    elif function_type in ["tanh"]:
        y_min = -1.2
    else:
        y_min = -0.2
    y_max = 1.2 if function_type in ["sigmoid", "tanh"] else 10
    ax.set_ylim(y_min, y_max)
    # vertical line at activation point until it hits the bias value
      # plot the neuron graph
    weight = slope
    z = weight * x + bias
    output = activation_function(z, function_type=function_type)
    ax.axvline(x=x, color="gray", linestyle='--', label=f"Input x={x:.2f}")

    ax.legend(fontsize=8, bbox_to_anchor=(1.4, 1))
    ax.set_title(f"Slope={slope:.2f}, Bias={bias:.2f}")

    ax1.set_aspect(0.5)
    g = Graph(
        [('x', 'out')],
        node_labels={'x': f'x={x:.1f}', 'out': f'out={output:.1f}'},
        edge_width={('x', 'out'): abs(weight) * 1},
        edge_labels={('x', 'out'): f'w={weight:.1f}'},
        node_layout={'x': (0.1, 0.5), 'out': (0.9, 0.5)},
        node_size=10,
        arrows=True,
        ax=ax1,
    )
    # Set font sizes directly on the artists
    for artist in g.node_label_artists.values():
        artist.set_fontsize(12)
    for artist in g.edge_label_artists.values():
        artist.set_fontsize(12)
    # Bias as text below neuron node — get the neuron position in display coords
    ax1.text(0.5, 0.3, f'b={bias:.2f}', ha='center', va='top',
             fontsize=12, color='#E67E22', transform=ax1.transData)
    #ax1.set_xlim(0, 1)

    fig.tight_layout();


def update_layer_of_neurons_using_slope_activation_point(\
    slope_1, slope_2, slope_3, \
    activation_point_1, activation_point_2, activation_point_3, \
    weight_1, weight_2, weight_3, \
    output_bias, function_type, sample_data):

    import matplotlib.pyplot as plt
    x_sample, y_sample, y_true = sample_data
    # calculate bias from activation points for each neuron
    bias_1 = -slope_1 * activation_point_1
    bias_2 = -slope_2 * activation_point_2
    bias_3 = -slope_3 * activation_point_3
    # combined output
    x_global = np.linspace(-10, 10, 1000)
    z_1_global = slope_1 * x_global + bias_1
    z_2_global = slope_2 * x_global + bias_2
    z_3_global = slope_3 * x_global + bias_3
    y_1_global = activation_function(z_1_global, function_type=function_type)
    y_2_global = activation_function(z_2_global, function_type=function_type)
    y_3_global = activation_function(z_3_global, function_type=function_type)

    combined_output = weight_1 * y_1_global + weight_2 * y_2_global + weight_3 * y_3_global + output_bias
    x_global_true = np.linspace(0, 1, 1000)
    y_1_global_true = activation_function(slope_1 * x_global_true + bias_1, function_type=function_type)
    y_2_global_true = activation_function(slope_2 * x_global_true + bias_2, function_type=function_type)
    y_3_global_true = activation_function(slope_3 * x_global_true + bias_3, function_type=function_type)
    combined_output_true = weight_1 * y_1_global_true + weight_2 * y_2_global_true + weight_3 * y_3_global_true + output_bias

    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    ax[0].plot(x_global, y_1_global, label=f"Neuron 1 Output")
    ax[0].set_title(f"Neuron 1")
    ax[0].set_xlabel('Input (x)')
    ax[0].set_ylabel('Output (y)')
    #ax[0].legend()
    ax[0].set_xlim(-5, 10)
    #ax[0].set_xticks(np.arange(-4, 11, 4))
    ax[0].set_ylim(-5, 10)
    #ax[0].set_yticks(np.arange(-5, 11, 5))
    ax[1].plot(x_global, y_2_global, label=f"Neuron 2 Output")
    ax[1].set_title(f"Neuron 2")
    ax[1].set_xlabel('Input (x)')
    # hide y axis for ax[1]
    ax[1].set_ylim(-5, 10)
    ax[1].set_yticks([])
    #ax[1].legend()
    ax[1].set_xlim(-5, 10)
    #ax[1].set_xticks(np.arange(-4, 11, 4))
    ax[2].plot(x_global, y_3_global, label=f"Neuron 3 Output")
    ax[2].set_title(f"Neuron 3")
    ax[2].set_xlabel('Input (x)')
    # hide y axis for ax[2]
    ax[2].set_ylim(-5, 10)
    ax[2].set_yticks([])
    #ax[2].legend()
    ax[2].set_xlim(-5, 10)
    #ax[2].set_xticks(np.arange(-4, 11, 4))
    ax[3].plot(x_global, combined_output, label=f"Combined Output")
    #ax[3].plot(x_true, y_true, color="black", label="True Function", linestyle='--', linewidth=2)
    ax[3].plot(x_sample, y_sample, color="red", label="Sampled Data", linestyle='None', marker='o')
    ax[3].set_title(f"Combined Output")
    ax[3].set_xlabel('Input (x)')
    # hide y axis for ax[3]
    ax[3].set_ylim(0, 10)
    ax[3].set_yticks([0, 5, 10])
    # legend outside the plot
    #ax[3].legend(loc='upper left', bbox_to_anchor=(1.2, 1))
    ax[3].set_xlim(0, 10)
    #ax[3].set_xticks(np.arange(0, 11, 2))

    # compute mean squared error between combined output and true function at the sampled points
    
    mean_squared_error = np.mean((combined_output_true - y_true)**2)
    plt.suptitle(f"Mean Squared Error: {mean_squared_error:.2f}")
    plt.tight_layout()


def plot_loss_landscape_with_state(loss_fn, output_vector, states=None, window_size=10, tangent=None, \
                                   show_legend=True, figsize=(5,3), limit_y_axis=True):
    import matplotlib.pyplot as plt 
    import matplotlib.cm as cm 
    # ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    y_pred_around_output = np.linspace(output_vector-window_size, output_vector+window_size, 1000)
    loss_values = [loss_fn(output_vector, y_pred) for y_pred in y_pred_around_output]
    plt.figure(figsize=figsize)
    plt.plot(y_pred_around_output, loss_values)
    plt.xlabel("Predicted values")
    plt.ylabel("Loss")

    # Add points
    if states is not None:
        colors = cm.get_cmap('viridis', len(states))
        for i, state in enumerate(list(states.values())):
            plt.scatter(state['prediction'], state['loss'], label=state['condition'], color=colors(i))

    if tangent is not None:
        # tangent = slope
        tangent_line = tangent['output_error_signal'] * y_pred_around_output + (tangent['loss'] - tangent['output_error_signal'] * tangent['prediction'])
        plt.plot(y_pred_around_output, tangent_line, color="gray", \
                 linestyle="--", label=r'$\delta_{out}$' + '={:.2f}'.format(tangent['output_error_signal']))
    
    min_values_to_plot = min(loss_values)*0.9 - 0.1 * (max(loss_values) - min(loss_values))
    max_values_to_plot = max(loss_values)*1.1 + 0.1 * (max(loss_values) - min(loss_values))
    if limit_y_axis:
        plt.ylim(min_values_to_plot, max_values_to_plot)
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
        df_layer["Activation Function"] = [neuron.function_type for neuron in layer.neurons]
        df_layer["Input Vector"] = [neuron.input_vector for neuron in layer.neurons]
        df_layer["Output"] = [neuron.forward() for neuron in layer.neurons]
        df = pd.concat([df, df_layer], ignore_index=True)
    
    # round the weights, bias, input vector and output to 2 decimal places for better display
    df["Weights"] = df["Weights"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Bias"] = df["Bias"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Input Vector"] = df["Input Vector"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    df["Output"] = df["Output"].apply(lambda x: np.round(x, 2) if isinstance(x, np.ndarray) else x)
    
    print(df)


## Util functions for module 3
def define_model(num_neurons, num_depth):
    import torch
    import torch.nn as nn
    layers = []
    input_size = 1
    for _ in range(num_depth):
        layers.append(nn.Linear(input_size, num_neurons))
        layers.append(nn.ReLU())
        input_size = num_neurons
    layers.append(nn.Linear(input_size, 1))
    model = nn.Sequential(*layers)
    return model
def preprocess_data(x):
    import torch
    x_tensor_1 = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    # standardize the data
    x_mean = x_tensor_1.mean()
    x_std = x_tensor_1.std()
    x_tensor = (x_tensor_1 - x_mean) / x_std
    return x_tensor, x_mean.detach().numpy(), x_std.detach().numpy()

def train_model(model, x_train, y_train, num_epochs=1000, learning_rate=0.001):
    import torch
    import torch.nn as nn
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model
def evaluate_model(model, x_eval):
    import torch
    x_eval_tensor = torch.tensor(x_eval, dtype=torch.float32).unsqueeze(1)
    y_pred = model(x_eval_tensor).detach().numpy().flatten()
    return y_pred 

def train_multiple_times(model, x_train, y_train, x_test, n_runs=15):
    # preprocess the training data
    x_train_tensor, x_sample_mean, x_sample_std = preprocess_data(x_train)
    y_train_tensor, y_sample_mean, y_sample_std = preprocess_data(y_train)
    all_predictions = []
    for run in range(n_runs):
        # Re-initialize the model weights
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        # Train the model
        trained_model = train_model(model, x_train_tensor, y_train_tensor)
        # Get predictions
        x_test_standardized = (x_test - x_sample_mean) / x_sample_std

        predictions = evaluate_model(trained_model, x_test_standardized)
        all_predictions.append(predictions * y_sample_std + y_sample_mean)
    return np.array(all_predictions)


### Train utils 

def relu_derivative(z):
    return np.where(z > 0, 1, 0).squeeze()

def sigmoid_derivative(z):
    sigmoid_output = activation_function(z, function_type='sigmoid')
    return (sigmoid_output * (1 - sigmoid_output)).squeeze()
def tanh_derivative(z):
    tanh_output = activation_function(z, function_type='tanh')
    return (1 - tanh_output ** 2).squeeze()
def linear_derivative(z):
    return np.ones_like(z).squeeze()

def activation_function_derivative(z, function_type):
    if function_type == 'relu':
        return relu_derivative(z)
    elif function_type == 'linear':
        return linear_derivative(z)
    elif function_type == 'sigmoid':
        return sigmoid_derivative(z)
    elif function_type == 'tanh':
        return tanh_derivative(z)
    else:
        raise ValueError("Unsupported activation function for derivative.")
    
