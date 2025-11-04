import pandas as pd, numpy as np, re, networkx as nx
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv("/home/gururaj/datasets/PROCESSED_CERT+ROLES/device_conv.csv")
df['anomaly_status'] = df['anomaly_status'].replace({2: 1, 3: 1})
QUOTE = r"['\u2018\u2019\"\u201C\u201D]"
def parse_query(q):
    if pd.isna(q): return {'activity': None, 'user': None, 'pc': None}
    q = str(q)
    am = re.search(rf"SELECT {QUOTE}(\w+){QUOTE}\s+AS\s+activity", q, re.IGNORECASE)
    um = re.search(rf"user\s*=\s*{QUOTE}([\w\d]+){QUOTE}", q, re.IGNORECASE)
    pm = re.search(rf"pc\s*=\s*{QUOTE}([\w\d\-]+){QUOTE}", q, re.IGNORECASE)
    return {'activity': am.group(1) if am else None, 'user': um.group(1) if um else None, 'pc': pm.group(1) if pm else None}

parsed = df['Query'].apply(parse_query)
df['activity'] = parsed.apply(lambda x: x['activity'])
df['user'] = parsed.apply(lambda x: x['user'])
df['pc'] = parsed.apply(lambda x: x['pc'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp', 'activity', 'user', 'pc']).sort_values(['Timestamp']).reset_index(drop=True)
df['label'] = df['anomaly_status'].apply(lambda x: 0 if x == 0 else 1)
df['activity_encoded'] = df['activity'].map({'Connect': 1, 'Disconnect': 0})
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['is_weekend'] = df['Timestamp'].dt.dayofweek >= 5
df['user_encoded'] = LabelEncoder().fit_transform(df['user'])
df['pc_encoded'] = LabelEncoder().fit_transform(df['pc'])
df['role_encoded'] = LabelEncoder().fit_transform(df['Role'].fillna("None"))

df['time_since_last'] = df.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)
df['rolling_10_count'] = df.groupby('user')['activity_encoded'].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)
df['rolling_100_count'] = df.groupby('user')['activity_encoded'].rolling(100, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)
n = len(df); train_end = int(n * 0.7); val_end = int(n * 0.8)
df_train = df.iloc[:train_end].copy()
df_val   = df.iloc[train_end:val_end].copy()
df_test  = df.iloc[val_end:].copy()

# Training features (No need to iterate one by one)
# df_train['time_since_last'] = df_train.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)
# df_train['rolling_10_count'] = df_train.groupby('user')['activity_encoded'].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)
# df_train['rolling_100_count'] = df_train.groupby('user')['activity_encoded'].rolling(100, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)

df_train['rolling_3_anomaly'] = df_train.groupby('user')['label'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_train['rolling_10_anomaly'] = df_train.groupby('user')['label'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_train['rolling_20_anomaly'] = df_train.groupby('user')['label'].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_train['anomaly_momentum'] = df_train['rolling_10_anomaly'] - df_train['rolling_20_anomaly']

df_train['user_gap_mean'] = df_train.groupby('user')['time_since_last'].transform('mean')
df_train['user_gap_std'] = df_train.groupby('user')['time_since_last'].transform('std').replace(0,1)
df_train['gap_zscore'] = (df_train['time_since_last'] - df_train['user_gap_mean']) / df_train['user_gap_std']

pc_freq = df_train['pc'].value_counts().to_dict()
df_train['rare_pc_flag'] = df_train['pc'].map(pc_freq) < 10

df_train['rare_hour_for_user'] = (
    df_train.groupby('user')['hour'].transform(lambda x: x.map(x.value_counts())) < 3
).astype(int)

# Graph Features
G = nx.Graph(); G.add_edges_from(df_train[['user', 'pc']].drop_duplicates().values.tolist())
user_degree = dict(G.degree(df_train['user'].unique()))
pc_degree = dict(G.degree(df_train['pc'].unique()))
df_train['user_degree'] = df_train['user'].map(user_degree).fillna(0)
df_train['pc_degree'] = df_train['pc'].map(pc_degree).fillna(0)

feature_cols = [
    'hour_sin', 'hour_cos', 'dayofweek',
    'user_degree', 'pc_degree',
    'activity_encoded', 'role_encoded',
    'time_since_last', 'rolling_10_count', 'rolling_100_count', 
    'is_after_hours', 'is_weekend',
    'rolling_3_anomaly', 'rolling_10_anomaly', 'rolling_20_anomaly', 'anomaly_momentum',
    'rare_pc_flag', 'rare_hour_for_user',
    'user_gap_mean', 'user_gap_std', 'gap_zscore',
]

# Initialize all feature columns for df_val
df_val['user_degree'] = 0.0
df_val['pc_degree'] = 0.0
df_val['time_since_last'] = 0.0
df_val['rolling_10_count'] = 0
df_val['rolling_100_count'] = 0
df_val['rolling_3_anomaly'] = 0.0
df_val['rolling_10_anomaly'] = 0.0
df_val['rolling_20_anomaly'] = 0.0
df_val['anomaly_momentum'] = 0.0
df_val['rare_pc_flag'] = 0
df_val['rare_hour_for_user'] = 0
df_val['user_gap_mean'] = 0.0
df_val['user_gap_std'] = 1.0
df_val['gap_zscore'] = 0.0

G_val = nx.Graph()
# Copy the training graph structure
G_val.add_edges_from(df_train[['user', 'pc']].drop_duplicates().values.tolist())

# df_val['time_since_last'] = df_val.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)
# df_val['rolling_10_count'] = df_val.groupby('user')['activity_encoded'].rolling(10, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)
# df_val['rolling_100_count'] = df_val.groupby('user')['activity_encoded'].rolling(100, min_periods=1).sum().reset_index(level=0, drop=True).shift(1)

df_val['rolling_3_anomaly'] = df_val.groupby('user')['label'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_val['rolling_10_anomaly'] = df_val.groupby('user')['label'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_val['rolling_20_anomaly'] = df_val.groupby('user')['label'].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)
df_val['anomaly_momentum'] = df_val['rolling_10_anomaly'] - df_val['rolling_20_anomaly']



for i in range(len(df_val)):
    user = df_val.iloc[i]['user']
    pc = df_val.iloc[i]['pc']
    hist = df_train[df_train['user'] == user]
    user_full = pd.concat([hist, df_val.iloc[[i]]]).sort_values('Timestamp')
    idx = df_val.index[i]
    df_val.at[idx, 'user_gap_mean'] = user_full['time_since_last'].expanding().mean().loc[idx]
    df_val.at[idx, 'user_gap_std'] = user_full['time_since_last'].expanding().std().replace(0, 1).loc[idx]
    df_val.at[idx, 'gap_zscore'] = (df_val.at[idx, 'time_since_last'] - df_val.at[idx, 'user_gap_mean']) / df_val.at[idx, 'user_gap_std']

    #update pc into pc frequency
    pc_freq[pc] = pc_freq.get(pc, 0) + 1
    df_val.at[idx, 'pc_freq'] = pc_freq.get(pc, 0)

    df_val.at[idx, 'rare_pc_flag'] = int(pc_freq.get(pc, 0) < 10)
    df_val.at[idx, 'rare_hour_for_user'] = int(user_full['hour'].value_counts().get(user_full.loc[idx, 'hour'], 0) < 3)

    G_val.add_edge(user, pc)
    df_val.at[idx, 'user_degree'] = G_val.degree(user)
    df_val.at[idx, 'pc_degree']   = G_val.degree(pc)
    
X_train, y_train = df_train[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False), df_train['label']
X_val, y_val = df_val[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False), df_val['label']
X_test, y_test = df_test[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).infer_objects(copy=False), df_test['label']
cat_idx = [feature_cols.index(f) for f in [
    'activity_encoded', 'role_encoded', 'rare_pc_flag', 'rare_hour_for_user', 'is_weekend', 'dayofweek'
]]

smotenc = SMOTENC(categorical_features=cat_idx, random_state=108, k_neighbors=5)
X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)

cb = CatBoostClassifier(iterations=1500, learning_rate=0.07, depth=7, eval_metric='AUC',
        loss_function='Logloss', random_seed=108, verbose=100, early_stopping_rounds=70)
rf = RandomForestClassifier(n_estimators=200, random_state=108)
ensemble = VotingClassifier([('catboost', cb), ('rf', rf)], voting='soft')
ensemble.fit(X_train_res, y_train_res)

y_val_pred_proba = ensemble.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_thr = thresholds[np.argmax(f1_scores)] if len(thresholds) else 0.5

G_test = nx.Graph()
G_test.add_edges_from(pd.concat([df_train, df_val])[['user', 'pc']].drop_duplicates().values.tolist())

# Initialize all feature columns for df_test
df_test['user_degree'] = 0.0
df_test['pc_degree'] = 0.0
df_test['time_since_last'] = 0.0
df_test['rolling_10_count'] = 0
df_test['rolling_100_count'] = 0
df_test['rolling_3_anomaly'] = 0.0
df_test['rolling_10_anomaly'] = 0.0
df_test['rolling_20_anomaly'] = 0.0
df_test['anomaly_momentum'] = 0.0
df_test['rare_pc_flag'] = 0
df_test['rare_hour_for_user'] = 0
df_test['user_gap_mean'] = 0.0
df_test['user_gap_std'] = 1.0
df_test['gap_zscore'] = 0.0

y_test_pred_proba = []

y_test_copy = y_test.copy()

for i in range(len(df_test)):
    user = df_test.iloc[i]['user']
    pc = df_test.iloc[i]['pc']
    hist = pd.concat([df_train, df_val])
    hist = hist[hist['user'] == user]
    user_full = pd.concat([hist, df_test.iloc[[i]]]).sort_values('Timestamp')
    idx = df_test.index[i]
    # df_test.at[idx, 'time_since_last'] = user_full['Timestamp'].diff().dt.total_seconds().fillna(0).loc[idx]
    for w in [3,10,20]:
        df_test.at[idx, f'rolling_{w}_anomaly'] = user_full['label'].rolling(w, min_periods=1).mean().shift(1).loc[idx]
    # for w in [10,100]:
    #     df_test.at[idx, f'rolling_{w}_count'] = user_full['activity'].rolling(w, min_periods=1).count().loc[idx]
    df_test.at[idx, 'user_gap_mean'] = user_full['time_since_last'].expanding().mean().loc[idx]
    df_test.at[idx, 'user_gap_std'] = user_full['time_since_last'].expanding().std().replace(0, 1).loc[idx]
    df_test.at[idx, 'gap_zscore'] = (df_test.at[idx, 'time_since_last'] - df_test.at[idx, 'user_gap_mean']) / df_test.at[idx, 'user_gap_std']
    df_test.at[idx, 'anomaly_momentum'] = df_test.at[idx, 'rolling_10_anomaly'] - df_test.at[idx, 'rolling_20_anomaly']

    #update pc into pc frequency
    pc_freq[pc] = pc_freq.get(pc, 0) + 1
    df_test.at[idx, 'rare_pc_flag'] = int(pc_freq.get(pc, 0) < 10)
    df_test.at[idx, 'rare_hour_for_user'] = int(user_full['hour'].value_counts().get(user_full.loc[idx, 'hour'], 0) < 3)

    G_test.add_edge(user, pc)
    df_test.at[idx, 'user_degree'] = G_test.degree(user)
    df_test.at[idx, 'pc_degree']   = G_test.degree(pc)

    prediction = ensemble.predict_proba(df_test.loc[[idx], feature_cols].replace([np.inf, -np.inf], 0).fillna(0))[:, 1]
    y_test_pred_proba.append(prediction[0])
    decision = prediction[0] >= best_thr
    df_test.at[idx, 'label'] = 1 if decision else 0

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

fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curve'); plt.legend(); plt.grid(alpha=0.3)
plt.savefig('roc_curve_noleak.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"ROC AUC Score: {roc_auc:.4f}")
