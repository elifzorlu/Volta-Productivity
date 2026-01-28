import pandas as pd
import numpy as np
import os

def generate_synthetic_data(days=365):
    dates = pd.date_range(start="2025-01-01", periods=days)
    
    # 1. Realistic Bounds (Clipping)
    sleep = np.random.normal(7, 1.5, days)
    sleep = np.clip(sleep, 3, 12)  # No one sleeps < 3h or > 12h
    
    caffeine = np.random.normal(150, 50, days)
    caffeine = np.clip(caffeine, 0, 500)  # Max 500mg (safety limit)

    screen_time = np.random.normal(6, 2, days)
    screen_time = np.clip(screen_time, 1, 16)
    
    # 2. Relationship: More sleep + caffeine - too much screen = Better Productivity
    # Added some noise so the model has to actually "learn"
    productivity = (sleep * 8) + (caffeine * 0.1) - (screen_time * 3) + np.random.normal(0, 5, days)
    productivity = np.clip(productivity, 0, 100) # Score out of 100
    
    df = pd.DataFrame({
        'date': dates, 
        'sleep_hours': sleep, 
        'caffeine_mg': caffeine, 
        'screen_time_hours': screen_time,
        'productivity_score': productivity
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/productivity_data.csv', index=False)
    print(f"✅ Generated {days} days of realistic behavioral data -> data/productivity_data.csv")

if __name__ == "__main__":
    generate_synthetic_data()
