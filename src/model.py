from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

def tune_isolation_forest(df, param_grid, random_state=42):
    X_train = df[df['Class'] == 0].drop('Class', axis=1)  # unsupervised training (on legit only)
    X_test = df.drop('Class', axis=1)
    y_true = df['Class']

    best_auc = 0
    best_model = None
    best_params = {}

    for n_estimators in param_grid['n_estimators']:
        for contamination in param_grid['contamination']:
            for max_features in param_grid['max_features']:
                model = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    max_features=max_features,
                    random_state=random_state
                )
                model.fit(X_train)
                scores = -model.decision_function(X_test)  # Negative because higher scores indicate anomalies
                auc = roc_auc_score(y_true, scores)

                if auc > best_auc:
                    best_auc = auc
                    best_model = model
                    best_params = {
                        'n_estimators': n_estimators,
                        'contamination': contamination,
                        'max_features': max_features
                    }

    return best_model, best_params, best_auc

def predict_anomalies(model, df):
    features = df.drop('Class', axis=1)
    scores = model.decision_function(features)
    predictions = model.predict(features)

    df_result = df.copy()
    df_result['AnomalyScore'] = scores
    df_result['AnomalyPrediction'] = predictions
    return df_result

