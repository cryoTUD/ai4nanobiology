import tiktoken
from src.lm_utils import LanguageModel
import torch.nn as nn
import torch


# Create batches of data for training using sequential sampling
def get_batch(data, batch_size, block_size):
    # Randomly select starting indices for the batch
    start_indices = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # Create input and target batches
    x = torch.stack([data[i:i+block_size] for i in start_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in start_indices])
    
    return x, y
def evaluate(model, data, block_size, vocab_size, loss):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(data) - block_size, block_size):
            x = data[i:i+block_size].unsqueeze(0)  # Add batch dimension
            y = data[i+1:i+block_size+1].unsqueeze(0)
            logits = model(x, causal=True)
            total_loss += loss(logits.view(-1, vocab_size), y.view(-1)).item()
    return total_loss / (len(data) // block_size)

def generate_text(model, encoder, start_text, max_new_tokens=50, device='cpu'):
    model.eval()
    tokens = encoder.encode(start_text)
    input_indices = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension and move to device
    generated_indices = model.generate(input_indices, max_new_tokens=max_new_tokens)
    generated_tokens = generated_indices.squeeze(0).tolist()  # Remove batch dimension and convert to list
    generated_text = encoder.decode(generated_tokens)
    print(generated_text)
    

# Initialize the tokenizer 
enc = tiktoken.encoding_for_model("gpt-4o")   

# Get the device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the text data
shakespeare_path = "tinyshakespeare.txt"
with open(shakespeare_path, "r") as f:
    text = f.read()
print(f"Length of text: {len(text)} characters")

# Tokenize the text 
tokens = enc.encode(text)
print(f"Number of tokens: {len(tokens)}")
unique_tokens = list(set(tokens))
print(f"Number of unique tokens: {len(unique_tokens)}")
vocabulary_size = enc.n_vocab
print(f"Vocabulary size: {vocabulary_size} tokens")

# Convert the list of tokens to a PyTorch tensor
data = torch.tensor(tokens, dtype=torch.long).to(device)
percent_training = 0.9
n = int(percent_training * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Training data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

# Initialize the language model
n_layers = 4 
n_heads = 4
d_input = 64
context_length = 128
batch_size = 16
d_model = 64

tinyLM = LanguageModel(
    n_layers=n_layers,
    n_head=n_heads,
    d_input=d_input,
    d_model=d_model,
    context_length=context_length,
    vocab_size=vocabulary_size,
    d_ff=d_model*4
).to(device)

print(f"Model initialized with {sum(p.numel() for p in tinyLM.parameters())} parameters.")
# Set up the training loop
learning_rate = 1e-3
max_epochs = 1000
optimizer = torch.optim.Adam(tinyLM.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss().to(device)
print("Starting training...")
for epoch in range(max_epochs):
    # Set the model to training mode
    tinyLM.train()
    # Get a batch of training data
    x_batch, y_batch = get_batch(train_data, batch_size, context_length)
    # Set gradients to zero 
    optimizer.zero_grad()
    # Forward pass, set causal=True for auto-regressive language modeling
    logits = tinyLM(x_batch, causal=True)
    # Compute the loss 
    train_loss = loss(logits.view(-1, vocabulary_size), y_batch.view(-1))
    # Backpropagation 
    train_loss.backward()
    # Update the model parameters
    optimizer.step()
    if epoch % 100 == 0:
        val_loss = evaluate(tinyLM, val_data, context_length, vocabulary_size, loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}")
        generate_text(tinyLM, enc, start_text="Hello", max_new_tokens=20, device=device)


# After training, we can generate some text to see how well the model has learned
prompt = "To be, or not"
generated_text = generate_text(tinyLM, enc, start_text=prompt, max_new_tokens=200, device=device)
print("Generated text:")
print("-" * 40)
print(generated_text)