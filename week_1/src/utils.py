import os 
import numpy as np
import dropbox

def upload_file_to_dropbox(filepath):
    ACCESS_TOKEN = "sl.u.AGbFJtTH3AqHPmNuTX82EuWsU7DQyD8eT1tpgexoZYoF8ypEjsB5ErpxCwGP1v7Ecw566LeCoWK2baKupsR8OwLV-MJ0nhiURqcj5Hw4ztrbUiyNPQymLBQweqHXZjpP84p4OCPtY03yTFg7walK7lwzKIliwG5dpFG0J9Ksu51e_i9TSkGrAdNBs11e0qzCEwnNL7tOl4-iIz0ofmS7RLc-TtAOYglHXx4GwSsg6WJEYfviqpGW0x-yrK3vjY_AggiMiYN2TwYN7UuNgaIJBzoMKB36SFjIHHqqAwNDwNsy7m_6VBUjJfACfJc7Nvt5ddMpkLnAkCvEutPVmf--ghJtSV92Pydmqr7abqamtoop1T3gjUz33x0-Ic52i-fYITft1NcRWTwxF8-3QTKrOX8ipyYgZRmiAhl32WehOKLY-wfdVnttubO3pLcFv5yFCdUPUWVjVOaL_FoYljwUZjp3uOgDVWurW-HQfSfEPKamm2Gu4U0X6_vX7sRwXnDKuxdKqzk9eNDxlYCfFtykrc9uaBAvasPj5oC4L4vVdebZtgzQtRmwb0Pcj2w0QW34EwmAy_QY4aApuKsYEHt2JBpKEYAH3C_Rz1Qioo_OfT-PVCG50BrfpmXNdFP6oaQU5knPTPI1zocucC1aoXA2fNE13mTHvEYIGZAaB1i3F2DzOXQ5twMESaH7AhL3twMT60SZt0TuXs3-OESgUFQV_SoqINPJZC7F4EgZAvep_gMJNT2NB9XfmYHH_C4ah0FQ_FyviiapAl_54O4XkAetGsXXwTr8oB_BFquwwtVUzxQIKADe5OYKvOuPtpJmYJ4-4n9aJp2ezUMplxQPo-LbTtod4TFAMoSote4boxfGgglHb_-dJ_LDdLzjJHNUWR-pocpFquO_MKEI7sDzzrZYVNuDk1pLMSRjptlOvBqpsGuJ8Ny3ocZrbNGMUCsLyQbGe4i9bysdEkPRlF6fcfB49YhbXEMfIhpSf8aCb-xjKRa6A2ZHKWKQBtV9LTHggm0uYqVFjYOZ8PCq3qTWEtrxnyt9XFSf4DbtG_XiCFg0rCxwZ16In1HvNfIbEPv-usWl2v2h6jX3lmCTVuH6HF77lWLMF_fc2ks-34QhR7PTxFU4Z30eMXfvsjlSiyjjXrRO4IgzLxQKmBCBXmPvlcdwZtdD365opN11bG767JEvNMAXcYZDCTWakvTp0qDyOETupkGq9jbL1fzBQiGQXWh8sdlwV0iRwwpmulJQx2-JNFxxWSdFJBrzmVME8eDd3jSFRCir4EaWaGgZtC2-TIWjkkOJ"
    dbx = dropbox.Dropbox(ACCESS_TOKEN)
    dropboxpath = "/App/NB4170_1"
    with open(filepath, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropboxpath,
            mode=dropbox.files.WriteMode.overwrite  # overwrite if exists
        )
    print(f"Uploaded: {filepath} → {dropboxpath}")

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
    ax[3].legend(loc='upper left', bbox_to_anchor=(1.2, 1))
    ax[3].set_xlim(0, 10)
    #ax[3].set_xticks(np.arange(0, 11, 2))

    # compute mean squared error between combined output and true function at the sampled points
    
    mean_squared_error = np.mean((combined_output_true - y_true)**2)
    plt.suptitle(f"Mean Squared Error: {mean_squared_error:.2f}")
    plt.tight_layout()


def plot_loss_landscape_with_state(loss_fn, output_vector, states=None, window_size=10, tangent=None, \
                                   show_legend=True, figsize=(5,3)):
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
        tangent_line = tangent['slope'] * y_pred_around_output + (tangent['loss'] - tangent['slope'] * tangent['prediction'])
        plt.plot(y_pred_around_output, tangent_line, color="gray", \
                 linestyle="--", label=r'$\delta_{out}$' + '={:.2f}'.format(tangent['slope']))
    
    min_values_to_plot = min(loss_values)*0.9 - 0.1 * (max(loss_values) - min(loss_values))
    max_values_to_plot = max(loss_values)*1.1 + 0.1 * (max(loss_values) - min(loss_values))
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
