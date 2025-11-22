import pandas as pd, numpy as np, re, networkx as nx
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv("/home/gururaj/datasets/PROCESSED_CERT+ROLES/file_conv.csv")
if 'anomaly_status' in df.columns:
    df['anomaly_status'] = df['anomaly_status'].replace({2: 1, 3: 1})

QUOTE = r"['\u2018\u2019\"\u201C\u201D]"
def parse_file_query(query):
    if pd.isna(query):
        return {'filename': None, 'user': None, 'pc': None}
    q = str(query)
    # Filename
    fname = re.search(rf"SELECT\s*{QUOTE}([\w\.\-]+){QUOTE}\s*AS\s*filename", q)
    # User
    user = re.search(rf"user\s*=\s*{QUOTE}([\w\d]+){QUOTE}", q)
    # PC
    pc = re.search(rf"pc\s*=\s*{QUOTE}([\w\d\-]+){QUOTE}", q)
    return {
        'filename': fname.group(1) if fname else None,
        'user': user.group(1) if user else None,
        'pc': pc.group(1) if pc else None
    }

parsed = df['Query'].apply(parse_file_query)
df['filename'] = parsed.apply(lambda x: x['filename'])
df['user'] = parsed.apply(lambda x: x['user'])
df['pc'] = parsed.apply(lambda x: x['pc'])

df['Timestamp'] = pd.to_datetime(df.get('date', df.get('Timestamp')), errors='coerce')
df['role'] = df.get('Role', df.get('role'))

#Count of timestamp not properly parsed
print("Invalid Timestamps:", df['Timestamp'].isna().sum())

df = df.dropna(subset=['Timestamp', 'user', 'pc', 'role', 'filename']).sort_values(['Timestamp']).reset_index(drop=True)
if 'anomaly_status' in df.columns:
    df['label'] = df['anomaly_status'].apply(lambda x: 0 if x == 0 else 1)

df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['is_weekend'] = df['Timestamp'].dt.dayofweek >= 5

df['user_encoded'] = LabelEncoder().fit_transform(df['user'])
df['pc_encoded'] = LabelEncoder().fit_transform(df['pc'])
df['filename_encoded'] = LabelEncoder().fit_transform(df['filename'].fillna("None"))
df['role_encoded'] = LabelEncoder().fit_transform(df['role'].fillna("None"))

df['time_since_last'] = df.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)

n = len(df); train_end = int(n * 0.7); val_end = int(n * 0.8)
df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

df_train['rolling_3_anomaly'] = df_train.groupby('user')['label'].rolling(3, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_train['rolling_10_anomaly'] = df_train.groupby('user')['label'].rolling(10, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_train['rolling_20_anomaly'] = df_train.groupby('user')['label'].rolling(20, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_train['rolling_50_anomaly'] = df_train.groupby('user')['label'].rolling(50, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_train['anomaly_momentum'] = df_train['rolling_10_anomaly'] - df_train['rolling_20_anomaly']

df_train['user_gap_mean'] = df_train.groupby('user')['time_since_last'].transform('mean')
df_train['user_gap_std'] = df_train.groupby('user')['time_since_last'].transform('std').replace(0,1)
df_train['gap_zscore'] = (df_train['time_since_last'] - df_train['user_gap_mean']) / df_train['user_gap_std']

filename_freq = df_train['filename'].value_counts().to_dict()
df_train['rare_filename_flag'] = df_train['filename'].map(filename_freq) < 10

df_train['rare_hour_for_user'] = (
    df_train.groupby('user')['hour'].transform(lambda x: x.map(x.value_counts())) < 3
).astype(int)

df_train['files_per_hour'] = df_train.groupby(['user', df_train['Timestamp'].dt.floor('h')])['filename'].transform('count')

# Graph Features
df_train['user_degree'] = 0.0
df_train['pc_degree'] = 0.0

feature_cols = [
    'user_encoded', 'pc_encoded', 'role_encoded', 'filename_encoded',
    'hour_sin', 'hour_cos', 'dayofweek',
    'user_degree', 'pc_degree',
    'time_since_last',
    'is_after_hours', 'is_weekend',
    'rolling_3_anomaly', 'rolling_10_anomaly', 'rolling_20_anomaly', 'rolling_50_anomaly', 'anomaly_momentum',
    'rare_filename_flag', 'rare_hour_for_user',
    'gap_zscore',
    'files_per_hour',
    ]

# Initialize all feature columns for df_val
df_val['user_degree'] = 0.0
df_val['pc_degree'] = 0.0
df_val['rolling_3_anomaly'] = 0.0
df_val['rolling_10_anomaly'] = 0.0
df_val['rolling_20_anomaly'] = 0.0
df_val['rolling_50_anomaly'] = 0.0
df_val['anomaly_momentum'] = 0.0
df_val['rare_filename_flag'] = 0
df_val['rare_hour_for_user'] = 0
df_val['user_gap_mean'] = 0.0
df_val['user_gap_std'] = 0.0
df_val['gap_zscore'] = 0.0
df_val['files_per_hour'] = 0

df_val['rolling_3_anomaly'] = df_val.groupby('user')['label'].rolling(3, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_val['rolling_10_anomaly'] = df_val.groupby('user')['label'].rolling(10, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_val['rolling_20_anomaly'] = df_val.groupby('user')['label'].rolling(20, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_val['rolling_50_anomaly'] = df_val.groupby('user')['label'].rolling(50, min_periods=1).reset_index(level=0, drop=True).shift(1)
df_val['anomaly_momentum'] = df_val['rolling_10_anomaly'] - df_val['rolling_20_anomaly']

df_val['files_per_hour'] = df_val.groupby(['user', df_val['Timestamp'].dt.floor('h')])['filename'].transform('count')

# Cache user histories for faster lookup
user_hist_train = {user: df_train[df_train['user'] == user] for user in df_val['user'].unique()}

for i in range(len(df_val)):
    row = df_val.iloc[i]
    user = row['user']
    filename = row['filename']
    hist = user_hist_train.get(user, pd.DataFrame())
    user_full = pd.concat([hist, df_val.iloc[[i]]]).sort_values('Timestamp')
    idx = df_val.index[i]
    df_val.at[idx, 'user_gap_mean'] = user_full['time_since_last'].expanding().mean().loc[idx]
    df_val.at[idx, 'user_gap_std'] = user_full['time_since_last'].expanding().std().replace(0, 1).loc[idx]
    df_val.at[idx, 'gap_zscore'] = (df_val.at[idx, 'time_since_last'] - df_val.at[idx, 'user_gap_mean']) / df_val.at[idx, 'user_gap_std']

    #update filename frequency
    filename_freq[filename] = filename_freq.get(filename, 0) + 1

    df_val.at[idx, 'rare_filename_flag'] = int(filename_freq.get(filename, 0) < 10)
    df_val.at[idx, 'rare_hour_for_user'] = int(user_full['hour'].value_counts().get(user_full.loc[idx, 'hour'], 0) < 3)

X_train, y_train = df_train[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False), df_train['label']
X_val, y_val = df_val[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False), df_val['label']
y_test = df_test['label']
cat_idx = [feature_cols.index(f) for f in [
    'user_encoded', 'pc_encoded', 'role_encoded', 'filename_encoded',
    'rare_filename_flag', 'rare_hour_for_user', 'is_weekend', 'dayofweek', 'is_after_hours', 
]]

# smotenc = SMOTENC(categorical_features=cat_idx, random_state=108, k_neighbors=5)
# X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)

cb = CatBoostClassifier(iterations=1500, learning_rate=0.07, depth=7, eval_metric='AUC',
        loss_function='Logloss', random_seed=108, verbose=100, early_stopping_rounds=70, auto_class_weights='Balanced', cat_features=cat_idx)
rf = RandomForestClassifier(n_estimators=200, random_state=108)
ensemble = VotingClassifier([('catboost', cb), ('rf', rf)], voting='soft')
ensemble.fit(X_train, y_train)

y_val_pred_proba = ensemble.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_thr = thresholds[np.argmax(f1_scores)] if len(thresholds) else 0.5

# Initialize all feature columns for df_test
df_test['user_degree'] = 0.0
df_test['pc_degree'] = 0.0
df_test['rolling_3_anomaly'] = 0.0
df_test['rolling_10_anomaly'] = 0.0
df_test['rolling_20_anomaly'] = 0.0
df_test['rolling_50_anomaly'] = 0.0
df_test['anomaly_momentum'] = 0.0
df_test['rare_filename_flag'] = 0
df_test['rare_hour_for_user'] = 0
df_test['user_gap_mean'] = 0.0
df_test['user_gap_std'] = 1.0
df_test['gap_zscore'] = 0.0
df_test['files_per_hour'] = 0

y_test_pred_proba = []

y_test_copy = y_test.copy()

# Cache combined train+val history for faster lookup
train_val_combined = pd.concat([df_train, df_val])
user_hist_trainval = {user: train_val_combined[train_val_combined['user'] == user] for user in df_test['user'].unique()}

# Pre-compute hour value counts for each user from history
user_hour_counts = {}
for user in df_test['user'].unique():
    hist = user_hist_trainval.get(user, pd.DataFrame())
    if not hist.empty:
        user_hour_counts[user] = hist['hour'].value_counts().to_dict()
    else:
        user_hour_counts[user] = {}

# Initialize dictionaries to track files per hour
user_hour_files = {}  # {user: {hour_timestamp: file_count}}

for user in df_test['user'].unique():
    hist = user_hist_trainval.get(user, pd.DataFrame())
    if not hist.empty:
        # Initialize files per hour from history
        user_hour_files[user] = {}
        for _, h_row in hist.iterrows():
            hour_ts = h_row['Timestamp'].floor('h')
            user_hour_files[user][hour_ts] = user_hour_files[user].get(hour_ts, 0) + 1
    else:
        user_hour_files[user] = {}

user_length = df['user'].nunique()
pc_length = df['pc'].nunique()
bool_map = np.zeros((user_length, pc_length), dtype=bool)

for i in range(len(df_test)):
    row = df_test.iloc[i]
    user = row['user']
    filename = row['filename']

    idx = df_test.index[i]

    if bool_map[df_test.at[idx, 'user_encoded'], df_test.at[idx, 'pc_encoded']] == True:
        bool_map[df_test.at[idx, 'user_encoded'], df_test.at[idx, 'pc_encoded']] = False
        y_test_pred_proba.append(1)
        continue

    hist = user_hist_trainval.get(user, pd.DataFrame())
    user_full = pd.concat([hist, df_test.iloc[[i]]]).sort_values('Timestamp')

    for w in [3,10,20,50]:
        df_test.at[idx, f'rolling_{w}_anomaly'] = user_full['label'].rolling(w, min_periods=1).shift(1).loc[idx]
    
    df_test.at[idx, 'user_gap_mean'] = user_full['time_since_last'].expanding().mean().loc[idx]
    df_test.at[idx, 'user_gap_std'] = user_full['time_since_last'].expanding().std().replace(0, 1).loc[idx]
    df_test.at[idx, 'gap_zscore'] = (df_test.at[idx, 'time_since_last'] - df_test.at[idx, 'user_gap_mean']) / df_test.at[idx, 'user_gap_std']
    df_test.at[idx, 'anomaly_momentum'] = df_test.at[idx, 'rolling_10_anomaly'] - df_test.at[idx, 'rolling_20_anomaly']

    #update filename frequency
    filename_freq[filename] = filename_freq.get(filename, 0) + 1
    df_test.at[idx, 'rare_filename_flag'] = int(filename_freq.get(filename, 0) < 10)
    
    # Use pre-computed hour counts
    current_hour = row['hour']
    hour_count = user_hour_counts[user].get(current_hour, 0)
    df_test.at[idx, 'rare_hour_for_user'] = int(hour_count < 3)
    # Update hour counts for next iteration
    user_hour_counts[user][current_hour] = hour_count + 1

    # Compute files_per_hour based on data up to this point
    hour_ts = row['Timestamp'].floor('h')
    files_in_hour = user_hour_files[user].get(hour_ts, 0)
    df_test.at[idx, 'files_per_hour'] = files_in_hour
    
    # Update files_per_hour for next iteration
    user_hour_files[user][hour_ts] = files_in_hour + 1

    prediction = ensemble.predict_proba(df_test.loc[[idx], feature_cols].replace([np.inf, -np.inf], 0).fillna(0))[:, 1]
    y_test_pred_proba.append(prediction[0])
    decision = prediction[0] >= best_thr
    if decision:
     df_test.at[idx, 'label'] = 1
    # Update bool_map for the current user and pc if predicted as anomaly
    if decision == True:
        bool_map[df_test.at[idx, 'user_encoded'], df_test.at[idx, 'pc_encoded']] = True

y_test_pred_proba = np.array(y_test_pred_proba)
y_test_pred = (y_test_pred_proba >= best_thr).astype(int)

print("==== Test Set Results ====")
print(f"Accuracy: {accuracy_score(y_test_copy, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test_copy, y_test_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test_copy, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test_copy, y_test_pred, zero_division=0):.4f}")
cb_importance = ensemble.named_estimators_['catboost'].get_feature_importance()
cb_df = pd.Series(cb_importance, index=feature_cols)
rf_importance = ensemble.named_estimators_['rf'].feature_importances_
rf_df = pd.Series(rf_importance, index=feature_cols)
avg_importance = (cb_df + rf_df) / 2
print("Top Ensemble Features:\n", avg_importance.sort_values(ascending=False).head(20))

#save test set results in json 
import json
from datetime import datetime

test_results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "accuracy": float(accuracy_score(y_test_copy, y_test_pred)),
        "precision": float(precision_score(y_test_copy, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test_copy, y_test_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test_copy, y_test_pred, zero_division=0))
    },
    "threshold": float(best_thr),
    "top_features": avg_importance.sort_values(ascending=False).head(20).to_dict()
}

with open('test_results_file.json', 'w') as f:
    json.dump(test_results, f, indent=4)

print("Test results saved to test_results_file.json")

fpr, tpr, _ = roc_curve(y_test_copy, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('roc_curve_file_noleak.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC AUC Score: {roc_auc:.4f}")
