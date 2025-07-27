from src import preprocess, model, evaluate, utils
import joblib, os

def main():
    df = preprocess.load_data()
    train_df, test_df, val_df = preprocess.create_strategic_splits(df)

    print(f"Train: {train_df.shape}, Fraud: {train_df['Class'].sum()}")
    print(f"Test : {test_df.shape}, Fraud: {test_df['Class'].sum()}")
    print(f"Val  : {val_df.shape}, Fraud: {val_df['Class'].sum()}")

    train_df = preprocess.scale_features(train_df)
    test_df = preprocess.scale_features(test_df)
    val_df = preprocess.scale_features(val_df)

    param_dict = {
        "n_estimators": [100, 150],
        "contamination": [0.01, 0.02],
        'max_features': [1.0, 0.8]
    }

    my_model, my_params, my_auc = model.tune_isolation_forest(train_df, param_dict)
    print("\n✅ my Params:", my_params)
    print("✅ Train ROC-AUC:", round(my_auc, 4))

    test_result = model.predict_anomalies(my_model, test_df)
    print("\n🧪 Test Set Evaluation")
    evaluate.evaluate_model(test_result, title_suffix="Test")

    val_result = model.predict_anomalies(my_model, val_df)
    print("\n🔒 Validation Set Evaluation")
    evaluate.evaluate_model(val_result, title_suffix="Validation")

    # Save predictions
    os.makedirs("results", exist_ok=True)
    test_result.to_csv("results/test_predictions.csv", index=False)
    val_result.to_csv("results/val_predictions.csv", index=False)

    # Save trained model
    os.makedirs("model", exist_ok=True)
    joblib.dump(my_model, "model/isolation_forest_model.joblib")

    # Plot score distributions
    utils.plot_score_distribution(test_result, title="Test")
    utils.plot_score_distribution(val_result, title="Validation")

if __name__ == "__main__":
    main()
