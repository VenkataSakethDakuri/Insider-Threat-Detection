import re
import json
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.under_sampling import RandomUnderSampler  


with open('user_device.json') as f:
    user_map = json.load(f)
with open('pc_device.json') as f:
    pc_map = json.load(f)
with open('role_device.json') as f:
    role_map = json.load(f)


QUOTE = r"['\u2018\u2019\"]"  # ASCII ' or Unicode ' ' or "
ACTIVITY_RE = re.compile(rf"SELECT\s+{QUOTE}(\w+){QUOTE}\s+AS\s+activity", re.IGNORECASE)
USER_RE = re.compile(rf"user\s*=\s*{QUOTE}([\w\d]+){QUOTE}", re.IGNORECASE)
PC_RE = re.compile(rf"pc\s*=\s*{QUOTE}([\w\d\-]+){QUOTE}", re.IGNORECASE)


count = 0


def extract_feature_vector(query_str, role):
    global count
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

def custom_downsampler(X_train, y_train):
    train_data = X_train.copy()
    train_data['anomaly_status'] = y_train

    
    majority_class = train_data[train_data['anomaly_status'] == 0]
    minority_classes = train_data[train_data['anomaly_status'] != 0] 

    majority_indices = majority_class.index.tolist()
    minority_indices = minority_classes.index.tolist()   

    
    minority_count = len(minority_classes)
    filtered_majority_indices = np.random.choice(majority_indices, size=minority_count, replace=False)


    balanced_indices = np.concatenate([filtered_majority_indices, minority_indices])
    sorted_balanced_indices = np.sort(balanced_indices)


    X_train_balanced = X_train.loc[sorted_balanced_indices].reset_index(drop=True)
    y_train_balanced = y_train.loc[sorted_balanced_indices].reset_index(drop=True)


    return X_train_balanced, y_train_balanced


csv_file = "/home/gururaj/datasets/PROCESSED_CERT+ROLES/device_conv.csv"
df = pd.read_csv(csv_file)


df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')


df['hour'] = df['Timestamp'].dt.hour.fillna(-1).astype(int)
df['day_of_week'] = df['Timestamp'].dt.dayofweek.fillna(-1).astype(int)
df['after_hours'] = ((df['hour'] < 9) | (df['hour'] > 17)).astype(int)
df['is_weekday'] = df['day_of_week'].apply(lambda x: 1 if x in [0, 1, 2, 3, 4] else 0)


feature_vectors_final = []
target = []


df['sinusoidal_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cosine_hour'] = np.cos(2 * np.pi * df['hour'] / 24)


for i, row in df.iterrows():
    vec = extract_feature_vector(row['Query'], row['Role'])
    vec += [
        row['day_of_week'],
        row['after_hours'],
        row['is_weekday'],
        row['sinusoidal_hour'],
        row['cosine_hour'],
    ]
    feature_vectors_final.append(vec)
    target.append(row['anomaly_status'])


print(count)


feature_names = [
    'user_encoded', 'pc_encoded', 'role_encoded', 'activity_encoded',
    'day_of_week', 
    'after_hours', 'is_weekday', 
    'sinusoidal_hour', 'cosine_hour'
]


X = pd.DataFrame(feature_vectors_final, columns=feature_names)
y = pd.Series(target, name='anomaly_status')


print(f"No. of 1s in y : {y.sum()} out of {len(y)} samples")
print("First index of y with 1:", y[y == 1].index[0])


print(X.head())
print(y.head())


categorical_features = [0, 1, 2, 3, 4, 5, 6] 


# 70/10/20 train/val/test split
train_idx = int(len(X) * 0.7)
val_idx = int(len(X) * 0.8)  


X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]


# Make all 2,3 as 1 in y
y_train = y_train.replace({2: 1, 3: 1})
y_val = y_val.replace({2: 1, 3: 1})
y_test = y_test.replace({2: 1, 3: 1})


print("Unique values in y_train:", y_train.unique())
print("Unique values in y_test:", y_test.unique())

print(f"Training set size before downsampling: {len(X_train)}")
print(f"Class distribution before downsampling: {y_train.value_counts().to_dict()}")


X_train_downsampled, y_train_downsampled = custom_downsampler(X_train, y_train)


model_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',   
    #'auto_class_weights': 'Balanced',
    'cat_features': categorical_features,
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'random_seed': 42,
    'verbose': False
}


model_params['task_type'] = 'CPU'  # Default to CPU


model = CatBoostClassifier(**model_params)


model.fit(
    X_train_downsampled, y_train_downsampled,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=100
)


y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)


auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)


print(f"\nTest Set Results:")
print(f"Accuracy: {(y_test == y_pred).mean():.4f}")
print(f"Precision: {( ( (y_test == 1) & (y_pred == 1) ).sum() ) / ( (y_pred == 1).sum() + 1e-10 ):.4f}")
print(f"Recall: {( ( (y_test == 1) & (y_pred == 1) ).sum() ) / ( (y_test == 1).sum() + 1e-10 ):.4f}")
print(f"AUC: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")

#With all features and downsampling to 100 times minority class size
# #Accuracy: 0.9913
# Precision: 0.0909
# Recall: 0.0029
# AUC: 0.9374
# F1 Score: 0.0056

# With all features and downsampling to minority class size
# Accuracy: 0.9905
# Precision: 0.0784
# Recall: 0.0117
# AUC: 0.8962
# F1 Score: 0.0203