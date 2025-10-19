import pandas as pd
import json

csv_file = "/home/gururaj/datasets/raw_csvs/device.csv"
df = pd.read_csv(csv_file)
print("length of dataset:", len(df))

columns = ['user', 'pc', 'role']
mappings = {}

for col in columns:
    unique_vals = sorted(df[col].unique())
    print(f"No. of unique values in {col}: {len(unique_vals)}")
    values2id = {val: idx for idx, val in enumerate(unique_vals)}
    mappings[col] = values2id
    with open(f'{col}_device.json', 'w') as f:
        json.dump(values2id, f, indent=2)

for col in columns:
    df[col] = df[col].map(mappings[col])

# activity_map = {'Connect': 1, 'Disconnect': 0}
# df['activity'] = df['activity'].map(activity_map)

# df.to_csv("device_encoded.csv", index=False)

print("Device mappings saved to JSON files and device_encoded.csv created.")
