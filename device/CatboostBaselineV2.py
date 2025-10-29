import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Map Labels
df = pd.read_csv("/home/gururaj/datasets/PROCESSED_CERT+ROLES/device_conv.csv")
df['anomaly_status'] = df['anomaly_status'].replace({2: 1, 3: 1})
QUOTE = r"['\u2018\u2019\"\u201C\u201D]"
def parse_query(query):
    if pd.isna(query): return {'activity': None, 'user': None, 'pc': None}
    q = str(query)
    am = re.search(rf"SELECT {QUOTE}(\w+){QUOTE}\s+AS\s+activity", q, re.IGNORECASE)
    um = re.search(rf"user\s*=\s*{QUOTE}([\w\d]+){QUOTE}", q, re.IGNORECASE)
    pm = re.search(rf"pc\s*=\s*{QUOTE}([\w\d\-]+){QUOTE}", q, re.IGNORECASE)
    return {'activity': am.group(1) if am else None, 'user': um.group(1) if um else None, 'pc': pm.group(1) if pm else None}
parsed = df['Query'].apply(parse_query)
df['activity'] = parsed.apply(lambda x: x['activity'])
df['user'] = parsed.apply(lambda x: x['user'])
df['pc'] = parsed.apply(lambda x: x['pc'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df.dropna(subset=['Timestamp', 'activity', 'user', 'pc']).reset_index(drop=True)
df['label'] = df['anomaly_status'].apply(lambda x: 0 if x == 0 else 1)

# 2. Encoding
df['activity_encoded'] = df['activity'].map({'Connect': 1, 'Disconnect': 0})
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

user_encoder = LabelEncoder(); pc_encoder = LabelEncoder(); role_encoder = LabelEncoder()
df['user_encoded'] = user_encoder.fit_transform(df['user'])
df['pc_encoded'] = pc_encoder.fit_transform(df['pc'])
df['role_encoded'] = role_encoder.fit_transform(df['Role'].fillna("None"))

# 3. Rolling Features
df['time_since_last'] = df.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)
df['rolling_10_count'] = df.groupby('user')['activity'].rolling(10, min_periods=1).count().reset_index(level=0, drop=True)
df['rolling_100_count'] = df.groupby('user')['activity'].rolling(100, min_periods=1).count().reset_index(level=0, drop=True)
df['rolling_10_anomaly'] = df.groupby('user')['label'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)

# User baseline drift
df['user_mean_hour'] = df.groupby('user')['hour'].transform('mean')
df['user_hour_zscore'] = (df['hour'] - df['user_mean_hour']) / (df.groupby('user')['hour'].transform('std').replace(0, 1))
df['user_mean_pc_degree'] = df.groupby('user')['pc_encoded'].transform('nunique')

# 4. Graph Features
G = nx.Graph(); G.add_edges_from(df[['user', 'pc']].drop_duplicates().values.tolist())
user_degree = dict(G.degree(df['user'].unique()))
pc_degree = dict(G.degree(df['pc'].unique()))
df['user_degree'] = df['user'].map(user_degree).fillna(0)
df['pc_degree'] = df['pc'].map(pc_degree).fillna(0)
user_centrality = nx.betweenness_centrality(G); pc_centrality = user_centrality
df['user_centrality'] = df['user'].map(user_centrality).fillna(0)
df['pc_centrality'] = df['pc'].map(pc_centrality).fillna(0)

# 5. Advanced features
df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['connect_ratio'] = df.groupby('user')['activity_encoded'].transform('mean')

# 6. Feature columns
feature_cols = [
    'hour_sin', 'hour_cos', 'dayofweek',
    'user_degree', 'pc_degree', 'user_centrality', 'pc_centrality',
    'activity_encoded', 'role_encoded',
    'time_since_last', 'rolling_10_count', 'rolling_100_count', 'rolling_10_anomaly',
    'user_hour_zscore', 'user_mean_pc_degree', 'is_after_hours', 'connect_ratio'
]
X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y = df['label']

# 7. Split/Balancing
n = len(X); train_end = int(n*0.7); val_end = int(n*0.8)
X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

categorical_features_indices = [feature_cols.index('activity_encoded'), feature_cols.index('role_encoded')]
smotenc = SMOTENC(categorical_features=categorical_features_indices, random_state=108, k_neighbors=5)
X_train_res, y_train_res = smotenc.fit_resample(X_train, y_train)

# 8. Ensemble Model (CatBoost + RF voting)
cb = CatBoostClassifier(iterations=1500, learning_rate=0.07, depth=7, eval_metric='AUC', loss_function='Logloss', random_seed=108, verbose=100, early_stopping_rounds=70)
rf = RandomForestClassifier(n_estimators=200, random_state=108)
ensemble = VotingClassifier([('catboost', cb), ('rf', rf)], voting='soft')

ensemble.fit(X_train_res, y_train_res)
y_val_pred_proba = ensemble.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_thr = thresholds[np.argmax(f1_scores)] if len(thresholds) else 0.5

# 9. Evaluate
y_test_pred_proba = ensemble.predict_proba(X_test)[:,1]
y_test_pred = (y_test_pred_proba >= best_thr).astype(int)
print("==== Test Set Results ====")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

# Get CatBoost feature importance
cb_importance = ensemble.named_estimators_['catboost'].get_feature_importance()
cb_df = pd.Series(cb_importance, index=feature_cols)

# Get RandomForest feature importance
rf_importance = ensemble.named_estimators_['rf'].feature_importances_
rf_df = pd.Series(rf_importance, index=feature_cols)

# 10. Ensemble Feature Importance
avg_importance = (cb_df + rf_df) / 2
print("Top Ensemble Features:\n", avg_importance.sort_values(ascending=False).head(10))
