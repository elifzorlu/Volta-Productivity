import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import time

# 1. THE MODEL (The "AI" part)
# Simple LSTM that takes a sequence of days and predicts the next day's focus score.
class ProductivityLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=2):
        super(ProductivityLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully Connected Layer (Regression output)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # careful about the DEVICE (CPU vs GPU)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 2. DATA LOADER (Simple mocked version for now)
def get_dummy_data(seq_length=5):
    # Generates random tensors to simulate: [Batch Size, Sequence Length, Features]
    # Features = [Sleep Hours, Caffeine Intake]
    X = torch.randn(100, seq_length, 2) 
    y = torch.randn(100, 1) # Target = Focus Score
    return X, y

# 3. TRAINING LOOP (The "Engineering" part)
def train_model(epochs, learning_rate, device_name):
    print(f"--- 🚀 INITIALIZING VOLTA TRAINING PIPELINE ---")
    
    # HARDWARE CHECK: Verify if CUDA is actually available if requested
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA requested but not available. Fallback to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    
    print(f"✅ Hardware Accelerator: [{device}]")
    
    # Initialize Model & Move to Hardware
    model = ProductivityLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load Data & Move to Hardware
    X_train, y_train = get_dummy_data()
    X_train, y_train = X_train.to(device), y_train.to(device) 
    
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    duration = time.time() - start_time
    print(f"\n✨ Training Complete using {device} in {duration:.2f} seconds.")

# 4. COMMAND LINE INTERFACE (CLI)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Volta: LSTM Training Pipeline')
    
    # This argument allows you to toggle hardware (Great for Apple Interviews)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Compute device to use for training (cpu or cuda)')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    args = parser.parse_args()
    
    train_model(args.epochs, args.lr, args.device)
