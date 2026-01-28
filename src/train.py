import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import time
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# --- 1. CONFIGURATION ---
FEATURES = ['sleep_hours', 'caffeine_mg', 'screen_time_hours']
TARGET = 'productivity_score'
SEQ_LENGTH = 7  # Look back at past week

# --- 2. DATA PIPELINE ---
def load_and_process_data(csv_path, device):
    print(f"📊 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Scale Data 
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES + [TARGET]] = scaler.fit_transform(df[FEATURES + [TARGET]])
    
    # Save scaler for inference later
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/scaler.pkl')
    
    # Create Sequences (Sliding Window)
    data = df_scaled[FEATURES + [TARGET]].values
    sequences = []
    targets = []
    
    for i in range(len(data) - SEQ_LENGTH):
        seq = data[i:i+SEQ_LENGTH, :-1]  # Input: Features only
        label = data[i+SEQ_LENGTH, -1]   # Target: Productivity
        sequences.append(seq)
        targets.append(label)
    
    # Convert to Tensors
    X = torch.FloatTensor(np.array(sequences)).to(device)
    y = torch.FloatTensor(np.array(targets)).view(-1, 1).to(device)
    
    # Chronological Split (No random shuffle for time-series!)
    train_size = int(len(X) * 0.8)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

# --- 3. MODEL ARCHITECTURE ---
class VoltaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(VoltaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Take last time-step

# --- 4. TRAINING LOOP ---
def train_pipeline(args):
    # Hardware Check
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        
    print(f"🚀 Training on device: {device}")
    
    # Load Data
    X_train, y_train, X_val, y_val = load_and_process_data('data/productivity_data.csv', device)
    
    # Init Model
    model = VoltaLSTM(input_dim=len(FEATURES)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    start_time = time.time()
    model.train()
    
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            val_loss = criterion(model(X_val), y_val)
            print(f"Epoch {epoch} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")
            
    # Save Model
    torch.save(model.state_dict(), 'artifacts/volta_model.pth')
    print(f"\n✨ Done! Model saved to artifacts/volta_model.pth ({time.time()-start_time:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()
    
    train_pipeline(args)
