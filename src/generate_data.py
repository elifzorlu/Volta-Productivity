# generate_data.py
import pandas as pd
import numpy as np

# Create fake "productivity" data (0-100 score)
# simulating focus hours, sleep, and caffeine intake
def generate_synthetic_data(days=365):
    dates = pd.date_range(start="2025-01-01", periods=days)
    sleep = np.random.normal(7, 1.5, days) # Mean 7h sleep
    caffeine = np.random.normal(150, 50, days) # mg caffeine
    
    # Fake relationship: More sleep + some caffeine = better focus
    focus_score = (sleep * 10) + (caffeine * 0.1) + np.random.normal(0, 5, days)
    
    df = pd.DataFrame({'date': dates, 'sleep': sleep, 'caffeine': caffeine, 'focus': focus_score})
    df.to_csv('data/dummy_productivity.csv', index=False)
    print("Synthetic data generated.")

if __name__ == "__main__":
    generate_synthetic_data()
