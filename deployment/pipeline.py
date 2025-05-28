"""
Automated Data Pipeline for Regular Updates

- Downloads latest datasets from source (Kaggle, Dropbox, etc.)
- Cleans and preprocesses data
- Saves processed files to datasets/ for dashboard use
"""

import os
import pandas as pd
from datetime import datetime

def download_data():
    # Example: Download from Dropbox
    url = "https://www.dropbox.com/s/yourfile.csv?dl=1"
    target = "datasets/updated_data.csv"
    print(f"Fetching latest dataset from {url}")
    df = pd.read_csv(url)
    df.to_csv(target, index=False)
    print(f"Saved to {target}")

def clean_data():
    df = pd.read_csv("datasets/updated_data.csv")
    # Example cleaning
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.to_csv("datasets/cleaned_data.csv", index=False)
    print("Cleaned data saved.")

def main():
    download_data()
    clean_data()
    print(f"Pipeline finished at {datetime.now()}")

if __name__ == "__main__":
    main()