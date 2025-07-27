import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_score_distribution(df, title=""):
    if 'AnomalyScore' not in df.columns or 'Class' not in df.columns:
        raise ValueError("Required columns 'AnomalyScore' and 'Class' not found.")
    
    os.makedirs("results/plots", exist_ok=True)

    sns.histplot(data=df, x='AnomalyScore', hue='Class', bins=50, kde=True, palette={0: 'blue', 1: 'red'})
    plt.title(f"Anomaly Score Distribution ({title})")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/score_distribution_{title or 'default'}.png", dpi=300, bbox_inches='tight')
    plt.close()
