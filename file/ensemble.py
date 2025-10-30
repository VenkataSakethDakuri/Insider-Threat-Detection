import pandas as pd
import numpy as np
import re
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv("/home/gururaj/datasets/PROCESSED_CERT+ROLES/file_conv.csv")

# 2. Parse Query string 
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

if 'anomaly_status' in df.columns:
    df['anomaly_status'] = df['anomaly_status'].replace({2: 1, 3: 1})
    df['label'] = df['anomaly_status'].astype(int)

df = df.dropna(subset=['Timestamp', 'user', 'pc', 'role', 'filename', 'label']).reset_index(drop=True)

# 3. Label encoding: user, pc, role, filename
user_encoder = LabelEncoder();     df['user_encoded']     = user_encoder.fit_transform(df['user'])
pc_encoder = LabelEncoder();       df['pc_encoded']       = pc_encoder.fit_transform(df['pc'])
role_encoder = LabelEncoder();     df['role_encoded']     = role_encoder.fit_transform(df['role'])
filename_encoder = LabelEncoder(); df['filename_encoded'] = filename_encoder.fit_transform(df['filename'])

# 4. Time-based features
df['hour'] = df['Timestamp'].dt.hour
df['dayofweek'] = df['Timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 5. Rolling features
df['time_since_last'] = df.groupby('user')['Timestamp'].diff().dt.total_seconds().fillna(0)
df['rolling_10_count']   = df.groupby('user')['filename'].rolling(10, min_periods=1).count().reset_index(level=0, drop=True)
df['rolling_100_count']  = df.groupby('user')['filename'].rolling(100, min_periods=1).count().reset_index(level=0, drop=True)
df['rolling_10_anomaly'] = df.groupby('user')['label'].rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
df['rolling_3_anomaly']  = df.groupby('user')['label'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
df['rolling_20_anomaly'] = df.groupby('user')['label'].rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
df['anomaly_momentum']   = df['rolling_10_anomaly'] - df['rolling_20_anomaly']

df['user_gap_mean'] = df.groupby('user')['time_since_last'].transform('mean')
df['user_gap_std']  = df.groupby('user')['time_since_last'].transform('std').replace(0,1)
df['gap_zscore'] = (df['time_since_last'] - df['user_gap_mean']) / df['user_gap_std']

fname_freq = df['filename'].value_counts()
df['rare_filename_flag'] = df['filename'].map(fname_freq) < 10

df['rare_hour_for_user'] = (
    df.groupby('user')['hour'].transform(lambda x: x.map(x.value_counts())) < 3
).astype(int)

df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)
df['is_weekend']     = df['Timestamp'].dt.dayofweek >= 5

# 6. Graph Features: user-pc
G = nx.Graph(); G.add_edges_from(df[['user', 'pc']].drop_duplicates().values.tolist())
user_degree = dict(G.degree(df['user'].unique()))
pc_degree   = dict(G.degree(df['pc'].unique()))
df['user_degree'] = df['user'].map(user_degree).fillna(0)
df['pc_degree']   = df['pc'].map(pc_degree).fillna(0)

# 7. Final feature columns 
feature_cols = [
    'hour_sin', 'hour_cos', 'dayofweek',
    'user_degree', 'pc_degree',
    'user_encoded', 'pc_encoded', 'role_encoded', 'filename_encoded',
    'time_since_last', 'rolling_10_count', 'rolling_100_count', 'rolling_10_anomaly',
    'is_after_hours', 'is_weekend',
    'rolling_3_anomaly', 'rolling_20_anomaly', 'anomaly_momentum',
    'rare_filename_flag', 'rare_hour_for_user',
    'user_gap_mean', 'user_gap_std', 'gap_zscore'
]
X = df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
y = df['label']

# 8. Split/Balancing/Model
n = len(X); train_end = int(n * 0.7); val_end = int(n * 0.8)
X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

categorical_features_indices = [feature_cols.index(f) for f in [
    'user_encoded', 'pc_encoded', 'role_encoded', 'filename_encoded', 'rare_filename_flag', 'rare_hour_for_user', 'is_weekend', 'dayofweek'
]]
smotenc = SMOTENC(categorical_features=categorical_features_indices, random_state=108, k_neighbors=5)
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

y_test_pred_proba = ensemble.predict_proba(X_test)[:,1]
y_test_pred = (y_test_pred_proba >= best_thr).astype(int)
print("==== Test Set Results ====")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred, zero_division=0):.4f}")

# 9. Feature Importances
cb_importance = ensemble.named_estimators_['catboost'].get_feature_importance()
cb_df = pd.Series(cb_importance, index=feature_cols)
rf_importance = ensemble.named_estimators_['rf'].feature_importances_
rf_df = pd.Series(rf_importance, index=feature_cols)
avg_importance = (cb_df + rf_df) / 2
print("Top Ensemble Features:\n", avg_importance.sort_values(ascending=False).head(20))

# 10. ROC AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (File Events)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve_file.png', dpi=300, bbox_inches='tight')
print("ROC curve saved as 'roc_curve_file.png'")
plt.show()
print(f"\nROC AUC Score: {roc_auc:.4f}")

