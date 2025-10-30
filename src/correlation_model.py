# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, roc_auc_score

# def prepare_features(merged_df):
#     # simple features: sentiment_score, leverage, size
#     df = merged_df.copy()
#     df['sentiment_score'] = df['Classification'].map({'Fear': -1, 'Greed': 1}).fillna(0)
#     df['target_win'] = (df['closedPnL'] > 0).astype(int)
#     features = df[['sentiment_score','leverage','size']].fillna(0)
#     target = df['target_win']
#     return features, target

# def train_logistic(features, target):
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
#     model = LogisticRegression(max_iter=200)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     return {
#         'model': model,
#         'accuracy': accuracy_score(y_test, preds),
#         'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
#     }


# src/correlation_model.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def prepare_features(merged_df):
    # simple features: sentiment_score, leverage, size
    df = merged_df.copy()
    df['sentiment_score'] = df.get('Classification', pd.Series()).map({'Fear': -1, 'Greed': 1}).fillna(0)
    df['target_win'] = (df.get('closedPnL', pd.Series()) > 0).astype(int)
    features = df[['sentiment_score','leverage','size']].fillna(0)
    target = df['target_win'].fillna(0).astype(int)
    return features, target

def train_logistic(features, target, test_size=0.2, random_state=42):
    """
    Trains a simple logistic regression. Performs safety checks and returns a dict.
    If there is not enough data or not enough target-class variety, returns {'error': '...'}.
    """
    n = features.shape[0]
    unique_targets = target.unique()
    if n < 10:
        return {'error': 'Not enough samples to train model (need >= 10).', 'n_samples': int(n)}
    if len(unique_targets) < 2:
        return {'error': 'Target has only one class (all wins or all losses). Need both classes to train.', 'unique_targets': unique_targets.tolist()}

    try:
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_state, stratify=target if len(unique_targets)>1 else None)
    except Exception as e:
        return {'error': f'train_test_split failed: {str(e)}'}

    try:
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        return {
            'model': model,
            'accuracy': float(accuracy_score(y_test, preds)),
            'roc_auc': float(roc_auc_score(y_test, proba)) if proba is not None and len(np.unique(y_test)) > 1 else None,
            'n_samples': int(n)
        }
    except Exception as e:
        return {'error': f'Model training failed: {str(e)}'}
