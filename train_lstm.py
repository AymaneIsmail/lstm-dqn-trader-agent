import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.utils import plot_model

from utils import create_lstm_sequences_by_ticker  # doit retourner X, y_price

MODEL_PATH = "models/lstm_stock_model_multi.keras"

# Charger et préparer les données
df = pd.read_csv("data/final_dataset.csv")
df = df.sort_values(['Ticker', 'Date'])

feature_cols = ['Open', 'High', 'Low', 'Close']
lookback = 30

X, y_price = create_lstm_sequences_by_ticker(df, feature_cols, target_col='Close', lookback=lookback)

# Créer y_trend (1 = hausse, 0 = baisse ou stable)
y_trend = (np.roll(y_price, -1) > y_price).astype(int)
y_trend = y_trend[:-1]
X = X[:-1]
y_price = y_price[:-1]

# Split
X_train, X_test, y_price_train, y_price_test, y_trend_train, y_trend_test = train_test_split(
    X, y_price, y_trend, test_size=0.2, random_state=42, shuffle=True
)

# Création ou chargement du modèle
if os.path.exists(MODEL_PATH):
    print(f"Chargement du modèle depuis {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    history = None
else:
    print("Entraînement d'un nouveau modèle multi-sortie...")

    input_layer = Input(shape=(lookback, X.shape[2]))

    x = LSTM(64, return_sequences=True)(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)

    # Sortie 1 : prédiction prix
    output_price = Dense(1, name="price_output")(x)

    # Sortie 2 : prédiction tendance
    output_trend = Dense(1, activation="sigmoid", name="trend_output")(x)

    model = Model(inputs=input_layer, outputs=[output_price, output_trend])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            "price_output": MeanSquaredError(),
            "trend_output": BinaryCrossentropy()
        },
        metrics={
            "price_output": [MeanAbsoluteError()],
            "trend_output": ["accuracy"]
        }
    )

    model.summary()

    history = model.fit(
        X_train,
        {"price_output": y_price_train, "trend_output": y_trend_train},
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Modèle sauvegardé sous '{MODEL_PATH}'")

# Prédictions
pred_price, pred_trend = model.predict(X_test)

# Affichage
for i in range(5):
    print(f"Prédit : {pred_price[i][0]:.4f} - Réel : {y_price_test[i]:.4f} | Tendance : {int(pred_trend[i][0] > 0.5)} - Réelle : {y_trend_test[i]}")

rmse = np.sqrt(mean_squared_error(y_price_test, pred_price))
mae = mean_absolute_error(y_price_test, pred_price)
trend_acc = accuracy_score(y_trend_test, (pred_trend > 0.5).astype(int))

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Tendance Accuracy: {trend_acc:.2%}")

# Courbe de perte
if history:
    plt.plot(history.history['price_output_loss'], label='Prix - train loss')
    plt.plot(history.history['val_price_output_loss'], label='Prix - val loss')
    plt.plot(history.history['trend_output_loss'], label='Trend - train loss')
    plt.plot(history.history['val_trend_output_loss'], label='Trend - val loss')
    plt.title("Loss par sortie")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    date_str = datetime.now().strftime("%Y-%d-%m")
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/loss_plot-{date_str}.png"
    plt.savefig(filename)
    plt.show()

    print(f"Courbe de perte sauvegardée sous '{filename}'")
