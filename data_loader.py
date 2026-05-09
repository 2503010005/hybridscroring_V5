import os
import pandas as pd

def load_dataset(path):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.concat([pd.read_csv(os.path.join(path, f)) for f in files])
    return df.dropna()
    