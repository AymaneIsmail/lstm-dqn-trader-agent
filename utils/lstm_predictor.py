# utils/lstm_predictor.py

import numpy as np
from tensorflow.keras.models import load_model

class LSTMPredictor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict_next(self, sequence: np.ndarray) -> float:
        """
        Prédit le prix suivant à partir d'une séquence [lookback, features]
        """
        sequence = np.expand_dims(sequence, axis=0)  # Devient [1, lookback, features]
        pred_price, _ = self.model.predict(sequence, verbose=0)
        return pred_price[0][0]
