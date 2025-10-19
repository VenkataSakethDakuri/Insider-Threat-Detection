import re
import json
import pandas as pd
from imblearn.over_sampling import SMOTENC

with open('user_device.json') as f:
    user_map = json.load(f)
with open('pc_device.json') as f:
    pc_map = json.load(f)
with open('role_device.json') as f:
    role_map = json.load(f)

def extract_feature_vector(query_str, role):
    activity_match = re.search(r"SELECT [‘'](\w+)[’'] AS activity", query_str)
    activity = activity_match.group(1) if activity_match else None
    activity_encoded = 1 if activity == "Connect" else 0 if activity == "Disconnect" else -1

    user_match = re.search(r"user=’([\w\d]+)’", query_str)
    user = user_match.group(1) if user_match else None
    user_encoded = user_map.get(user, -1)

    pc_match = re.search(r"pc=’([\w\d\-]+)’", query_str)
    pc = pc_match.group(1) if pc_match else None
    pc_encoded = pc_map.get(pc, -1)

    # Role encoding from dataframe column
    role_encoded = role_map.get(role, -1)

    return [user_encoded, pc_encoded, role_encoded, activity_encoded]

csv_file = "datasets/raw_csvs/device_conv.csv"
df = pd.read_csv(csv_file)


df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

df['hour'] = df['Timestamp'].dt.hour.fillna(-1).astype(int)
df['day_of_week'] = df['Timestamp'].dt.dayofweek.fillna(-1).astype(int)
df['after_hours'] = ((df['hour'] < 9) | (df['hour'] > 17)).astype(int)
df['is_weekday'] = df['day_of_week'].apply(lambda x: 1 if x in [0, 1, 2, 3, 4] else 0)


feature_vectors_final = []
target = []

for i, row in df.iterrows():
    vec = extract_feature_vector(row['Query'], row['Role'])
    vec += [
        row['hour'],
        row['day_of_week'],
        row['after_hours'],
        row['is_weekday'],
    ]
    feature_vectors_final.append(vec)
    target.append(row['anomaly_status'])


feature_names = [
    'user_encoded', 'pc_encoded', 'role_encoded', 'activity_encoded',
    'hour', 'day_of_week', 'after_hours', 'is_weekday'
]
X = pd.DataFrame(feature_vectors_final, columns=feature_names)
y = pd.Series(target, name='anomaly_status')

print(X.head())
print(y.head())
