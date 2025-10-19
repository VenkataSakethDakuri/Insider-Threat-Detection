import re
import json
import pandas as pd
from imblearn.over_sampling import SMOTENC
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score

with open('user_device.json') as f:
    user_map = json.load(f)
with open('pc_device.json') as f:
    pc_map = json.load(f)
with open('role_device.json') as f:
    role_map = json.load(f)

QUOTE = r"['\u2018\u2019\"]"  # ASCII ' or Unicode ‘ ’ or "
ACTIVITY_RE = re.compile(rf"SELECT\s+{QUOTE}(\w+){QUOTE}\s+AS\s+activity", re.IGNORECASE)
USER_RE = re.compile(rf"user\s*=\s*{QUOTE}([\w\d]+){QUOTE}", re.IGNORECASE)
PC_RE = re.compile(rf"pc\s*=\s*{QUOTE}([\w\d\-]+){QUOTE}", re.IGNORECASE)

count = 0

def extract_feature_vector(query_str, role):
    activity_match = ACTIVITY_RE.search(query_str or "")
    activity = activity_match.group(1) if activity_match else None
    activity_encoded = 1 if activity == "Connect" else 0 if activity == "Disconnect" else -1

    user_match = USER_RE.search(query_str or "")
    user = user_match.group(1) if user_match else None
    user_encoded = user_map.get(user, -1)

    pc_match = PC_RE.search(query_str or "")
    pc = pc_match.group(1) if pc_match else None
    pc_encoded = pc_map.get(pc, -1)

    role_encoded = role_map.get(role, -1)
    count += 1
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

print(count)

feature_names = [
    'user_encoded', 'pc_encoded', 'role_encoded', 'activity_encoded',
    'hour', 'day_of_week', 'after_hours', 'is_weekday'
]
X = pd.DataFrame(feature_vectors_final, columns=feature_names)
y = pd.Series(target, name='anomaly_status')

print(X.head())
print(y.head())

categorical_features = [0, 1, 2, 3, 4, 5, 6, 7] 

tscv = TimeSeriesSplit(n_splits=5)  # growing/expanding training window

fold_stats = []
for fold, (tr, va) in enumerate(tscv.split(X), 1):
    X_tr, y_tr = X.iloc[tr], y.iloc[tr]
    X_va, y_va = X.iloc[va], y.iloc[va]

    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',  # handle imbalance automatically
        cat_features=categorical_features,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=100
    )

    p = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    f1 = f1_score(y_va, (p >= 0.5).astype(int))
    fold_stats.append((fold, auc, f1))
    print(f"Fold {fold}: AUC={auc:.4f}, F1={f1:.4f}")