import torch
import joblib
import pandas as pd
import numpy as np
import argparse
from train import VoltaLSTM, FEATURES, SEQ_LENGTH

def predict_next_day(model_path, scaler_path, data_path):
    # 1. Load Artifacts
    model = VoltaLSTM(input_dim=len(FEATURES))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    
    # 2. Get Last 7 Days of Data
    df = pd.read_csv(data_path)
    last_week = df[FEATURES].tail(SEQ_LENGTH).values
    
    # 3. Scale Input (Must match training scale)
    # Note: We need a dummy target column to use the same scaler
    dummy_input = np.hstack([last_week, np.zeros((SEQ_LENGTH, 1))]) 
    scaled_input = scaler.transform(dummy_input)[:, :-1] # Drop target col
    
    # 4. Inference
    tensor_in = torch.FloatTensor(scaled_input).unsqueeze(0) # Add batch dim
    with torch.no_grad():
        prediction_scaled = model(tensor_in).item()
        
    # 5. Inverse Scale to get real Score
    # Construct dummy row for inverse transform
    dummy_output = np.zeros((1, len(FEATURES) + 1))
    dummy_output[0, -1] = prediction_scaled
    prediction_real = scaler.inverse_transform(dummy_output)[0, -1]
    
    print(f"🔮 Based on the last {SEQ_LENGTH} days of behavior...")
    print(f"Predicted Productivity Score: {prediction_real:.2f} / 100")

if __name__ == "__main__":
    predict_next_day('artifacts/volta_model.pth', 'artifacts/scaler.pkl', 'data/productivity_data.csv')
