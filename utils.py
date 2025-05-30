import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def convert_to_datetime(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned

def min_max_normalizer(df: pd.DataFrame) -> pd.DataFrame:
    prices_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    
    df_normalized = df.copy()
    
    for col in prices_columns:
        if col in df_normalized.columns:
            scaler = MinMaxScaler()
            
            col_data = df_normalized[[col]]
            normalized_col = scaler.fit_transform(col_data)
            
            df_normalized[col] = normalized_col.flatten()
    
    return df_normalized


def create_lstm_sequences_by_ticker(df, feature_cols, target_col='Close', lookback=30):
    """
    Crée des séquences LSTM à partir d'un DataFrame multi-tickers.

    Args:
        df (pd.DataFrame): DataFrame contenant les données normalisées triées.
        feature_cols (list): colonnes utilisées comme entrées (X).
        target_col (str): colonne cible (y).
        lookback (int): taille de la fenêtre temporelle.

    Returns:
        X (np.array): séquences d'entrée [samples, lookback, features]
        y (np.array): cibles correspondantes [samples]
    """
    X_all, y_all = [], []

    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values('Date')

        if len(group) <= lookback:
            continue  # Pas assez de données pour ce ticker

        data = group[feature_cols + [target_col]].values

        for i in range(len(data) - lookback):
            X_seq = data[i:i+lookback, :-1]  # toutes les colonnes sauf target
            y_seq = data[i+lookback, -1]     # target à t+1
            X_all.append(X_seq)
            y_all.append(y_seq)

    return np.array(X_all), np.array(y_all)
