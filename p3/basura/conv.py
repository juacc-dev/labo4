import pandas as pd
from pathlib import Path


data_dir = Path("./data/2 - 22-10")

for file in data_dir.glob("*.csv"):
    df = pd.read_csv(file)
    freq = df["Frecuencia [Hz]"]

    if freq[1] - freq[0] > 0:
        continue

    print(f"Found {file.stem}")

    df = df[::-1]

    df.to_csv(file, index=False)
