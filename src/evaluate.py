import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(df_result, title_suffix=""):
    os.makedirs("results/plots", exist_ok=True)

    y_true = df_result['Class']
    y_pred = df_result['AnomalyPrediction'].apply(lambda x: 1 if x == -1 else 0)
    y_score = -df_result['AnomalyScore']

    auc = roc_auc_score(y_true, y_score)
    print(f"\nROC-AUC Score ({title_suffix}): {auc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix ({title_suffix}):")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({title_suffix})")
    plt.tight_layout()
    plt.savefig(f"results/plots/confusion_matrix_{title_suffix or 'default'}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({title_suffix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/roc_curve_{title_suffix or 'default'}.png", dpi=300, bbox_inches='tight')
    plt.close()
