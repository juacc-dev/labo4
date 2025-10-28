import pandas as pd
from pathlib import Path


data_dir = Path("./data/1 - 24-9")

for file in data_dir.glob("*.csv"):
    print(f"Found {file.stem}")
    df = pd.read_csv(file)

    df["Ruido X [V]"] = 5e-6
    df["Ruido Y [V]"] = 5e-6

    df.to_csv(file, index=False)
