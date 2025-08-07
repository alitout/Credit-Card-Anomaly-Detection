import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, RANDOM_STATE, SAMPLE_SIZE

def load_data():
    return pd.read_csv(DATA_PATH)


def create_strategic_splits(df, total_sample_size=SAMPLE_SIZE, val_fraud_size=93, train_ratio=0.8, random_state=RANDOM_STATE):
    # Split data
    fraud_df = df[df['Class'] == 1].sample(frac=1, random_state=random_state)
    fraud_val = fraud_df[:val_fraud_size]  # held out
    fraud_train_test = fraud_df[val_fraud_size:val_fraud_size + 400]

    # Sample legit data to match total 5000 size
    legit_df = df[df['Class'] == 0]
    legit_needed = total_sample_size - len(fraud_train_test)  # 5000 - 400 = 4600
    legit_sample = legit_df.sample(n=legit_needed, random_state=random_state)

    # Combine for main set
    full_df = pd.concat([fraud_train_test, legit_sample]).sample(frac=1, random_state=random_state)

    # Split into train/test
    train_df, test_df = train_test_split(
        full_df, test_size=(1 - train_ratio), stratify=full_df['Class'], random_state=random_state
    )

    # Validation set from remaining fraud and legit
    legit_val = legit_df.drop(legit_sample.index).sample(n=400, random_state=random_state)
    val_df = pd.concat([fraud_val, legit_val]).sample(frac=1, random_state=random_state)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), val_df.reset_index(drop=True)


def scale_features(df):
    # Scale 'Amount' and 'Time' features
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)
    return df
