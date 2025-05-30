from utils import clean_dataset, min_max_normalizer, convert_to_datetime
import pandas as pd
import glob
import os

DATASET_PATH = "data/*.csv"
OUTPUT_PATH = "data/final_dataset.csv"

csv_files = glob.glob(DATASET_PATH)

processed_dfs = []

print("📊 Statistiques volume par ticker dans chaque fichier:")

for file in csv_files:
    print(f"\nTraitement du fichier : {file}")
    
    df = pd.read_csv(file)
    df = convert_to_datetime(df, 'Date')
    df_cleaned = clean_dataset(df)
    
    # Affichage stats volume par ticker dans ce fichier
    for ticker in df_cleaned['Ticker'].unique():
        volumes = df_cleaned[df_cleaned['Ticker'] == ticker]['Volume']
        print(f"Ticker: {ticker}")
        print(f"  Volume min: {volumes.min()}")
        print(f"  Volume max: {volumes.max()}")
        print(f"  Volume mean: {volumes.mean():.2f}")
        print(f"  Volume median: {volumes.median()}")
        print("-" * 30)
    
    df_normalized = min_max_normalizer(df_cleaned)
    processed_dfs.append(df_normalized)

df_final = pd.concat(processed_dfs, ignore_index=True)
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Fichier final sauvegardé sous : {OUTPUT_PATH}")
print(f"\n🧾 Aperçu du fichier final :\n{df_final.head()}")
