import numpy as np
def define_model():
    import torch
    import torch.nn as nn

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

def train_model(model, criterion, optimizer, num_epochs, list_of_next_moves, batch_size):
    import torch
    # Create training data generator
    x_input_list = []
    y_output_list = []
    for game in list_of_next_moves:
        for step in game:
            x_input = step[0]
            y_output = step[1]
            x_input_list.append(x_input)
            y_output_list.append(y_output)


    x_input_tensor = torch.tensor(np.array(x_input_list), dtype=torch.float32).to(device)
    y_output_tensor = torch.tensor(np.array(y_output_list), dtype=torch.long).to(device)


    # create dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x_input_tensor, y_output_tensor)

    # split data into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)    

    # Training loop
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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
    return model



