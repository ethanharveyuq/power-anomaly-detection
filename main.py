import sys
import os

# Force Python to see your root project folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.load_data import load_data
from src.process_data import process_data

def run():
    print("--- Script Started ---")  # Visual proof that the script is running
    
    df = load_data('data/Bd18850_2021100102.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    df, mask = process_data(df)
    print(f"Data processed. Mask count: {mask.sum()}")
    
    print("--- Printing Results ---")
    print(df[mask])

if __name__ == "__main__":
    run()
