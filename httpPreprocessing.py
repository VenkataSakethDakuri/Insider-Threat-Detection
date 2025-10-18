import pandas as pd
import json

csv_file = "datasets/raw_csvs/http.csv"
df = pd.read_csv(csv_file)

columns = ['user', 'pc', 'role', 'url']
mappings = {}

for col in columns:
    unique_vals = sorted(df[col].unique())
    values2id = {val: idx for idx, val in enumerate(unique_vals)}
    mappings[col] = values2id
    with open(f'{col}_http.json', 'w') as f:
        json.dump(values2id, f, indent=2)

for col in columns:
    df[col] = df[col].map(mappings[col])

df.to_csv("http_encoded.csv", index=False)

print("HTTP mappings saved to JSON files and http_encoded.csv created.")
