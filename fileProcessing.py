import pandas as pd
import json

csv_file = "datasets/raw_csvs/file.csv"
df = pd.read_csv(csv_file)

columns = ['user', 'pc', 'role', 'filename']
mappings = {}

for col in columns:
    unique_vals = sorted(df[col].unique())
    values2id = {val: idx for idx, val in enumerate(unique_vals)}
    mappings[col] = values2id
    with open(f'{col}_file.json', 'w') as f:
        json.dump(values2id, f, indent=2)

for col in columns:
    df[col] = df[col].map(mappings[col])

df.to_csv("file_encoded.csv", index=False)

print("File mappings saved to JSON files and file_encoded.csv created.")
