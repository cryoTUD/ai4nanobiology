import numpy as np

def load_model_from_path(model_path):
    import torch 
    device = torch.device("cpu")
    model = define_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def save_model(model, model_name, save_folder="src/models/retrained_models/"):
    import torch 
    import os 
    os.makedirs(save_folder, exist_ok=True)
    if model_name.endswith(".pt"):
        new_model_name = model_name
    else:
        new_model_name = model_name + ".pt"

    model_path = os.path.join(save_folder, new_model_name)
    if os.path.exists(model_path):
        print(f"Model with name {new_model_name} already exists. Overwriting.")
    torch.save(model.state_dict(), model_path)
    return model_path

def define_model(reproducible=False):
    import torch
    import torch.nn as nn
    # set random seed for reproducibility
    if reproducible:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    device = torch.device("cpu")

    class LegalMovesModel(nn.Module):
        def __init__(self):
            super(LegalMovesModel, self).__init__()
            self.fc1 = nn.Linear(9, 81)
            self.fc2b = nn.Linear(81, 81)
            self.fc3 = nn.Linear(81, 9)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
            self.dropout = nn.Dropout(p=0.2)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            #x = self.relu(self.fc2(x))
            #x = self.dropout(x)
            #x = self.relu(self.fc2a(x))
            #x = self.dropout(x)
            x = self.relu(self.fc2b(x))
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x
    model = LegalMovesModel().to(device)
    return model

def train_model(model, x_input_list, y_output_list, num_epochs, batch_size=32, lr=0.001, weight_decay=1e-5, verbose="low"):
    import torch
    import torch.nn as nn   
    from torch.utils.data import TensorDataset, DataLoader
    from tqdm import tqdm
    # set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    device = torch.device("cpu")
    # Create training data generator
    x_input_tensor = torch.tensor(np.array(x_input_list), dtype=torch.float32).to(device)
    y_output_tensor = torch.tensor(np.array(y_output_list), dtype=torch.long).to(device)

    # create dataset and dataloader    
    dataset = TensorDataset(x_input_tensor, y_output_tensor)

    # split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)    

    # Training loop
    if verbose == "low":
        progress_bar = tqdm(range(num_epochs), desc="Training in progress")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataset)
        running_val_loss = 0.0
        for inputs, labels in val_loader:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                running_val_loss += val_loss.item() * inputs.size(0)
        epoch_val_loss = running_val_loss / len(val_dataset)
        if verbose == "low":
            # update progress bar just show show percentage progress
            progress_bar.update(1)
        elif verbose == "detail":
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        else:
            pass

    return model

def generate_data_from_user_input(n_games=15, username="username", opponent="random"):
    from .game_utils import User_vs_Computer
    import random
    from IPython.display import clear_output
    from time import sleep
    # generate training data from user input from n_games games
    game_data = {}
    player_1 = username
    player_2_name = random.choice(["Alice", "Bob"]) if opponent == "random" else opponent
    for i, _ in enumerate(range(n_games)):
        data_to_use = []
        result = User_vs_Computer(username, player_2_name)
        user_inputs = result["all_user_inputs"]
        for move in user_inputs[1:]: # skip the first move since it is always the same
            game_state = move[0]
            current_player = move[1]
            move_made = move[2]
            if current_player == 1:
                x_input = game_state
                y_output = move_made
                data_to_use.append((x_input, y_output))
        
        sleep(2)
        clear_output(wait=True)
        
        game_data[f"game_{i+1}"] = data_to_use
    return game_data

def retrain_using_generated_data(generated_data, strategy="train_alice"):
    '''
    Retrain the model using the generated data.
    strategy: "fresh_start" or "train_alice" or "train_bob" or path_to_model.pt
    '''
    import os 
    from src.train_game_utils import load_model_from_path, train_model, define_model

    # From the generated data, split into X and Y 
    X_input = []
    Y_output = []

    for game in generated_data.keys():
        user_input = generated_data[game]
        for move in user_input:
            X_input.append(move[0])
            Y_output.append(move[1])

    if strategy == "train_alice":
        model = load_model_from_path("src/models/Alice.pt")
    elif strategy == "train_bob":
        model = load_model_from_path("src/models/Bob.pt")
    elif strategy == "fresh_start":
        model = define_model()
    elif strategy.endswith(".pt"):
        model = load_model_from_path(strategy)
    else:
        raise ValueError("Invalid strategy. Choose from 'fresh_start', 'train_alice', 'train_bob', or provide a path to a .pt model file.")
    
    # Train the model on the generated data
    retrained = train_model(model, X_input, Y_output, num_epochs=100, verbose="low")

    return retrained




        
        

    

