import os
import joblib
import logging
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/intent_model.pkl"
DATA_PATH = "nlu/data/processed_intents.csv"


class IntentClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None

    # ---------------------------
    # Load Dataset
    # ---------------------------
    def load_data(self, path: str = DATA_PATH) -> Tuple[pd.Series, pd.Series]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        if "pattern" not in df.columns or "intent" not in df.columns:
            raise KeyError(
                "CSV must contain 'pattern' (text) and 'intent' columns."
            )

        return df["pattern"], df["intent"]

    # ---------------------------
    # Train Model
    # ---------------------------
    def train(self):
        logger.info("Loading dataset...")
        X, y = self.load_data()

        logger.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Building training pipeline...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=200))
        ])

        logger.info("Training model...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating model...")
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, preds))

        logger.info("Saving model...")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, self.model_path)

        logger.info(f"Model saved at {self.model_path}")

    # ---------------------------
    # Load Saved Model
    # ---------------------------
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model not found. Train the model first.")
        self.model = joblib.load(self.model_path)
        logger.info("Intent model loaded.")

    # ---------------------------
    # Predict Intent
    # ---------------------------
    def predict(self, text: str):
        if self.model is None:
            self.load()
        return self.model.predict([text])[0]

    # ---------------------------
    # Predict with Confidence
    # ---------------------------
    def predict_proba(self, text: str):
        if self.model is None:
            self.load()

        probs = self.model.predict_proba([text])[0]
        intent = self.model.predict([text])[0]
        confidence = max(probs)

        return intent, confidence


# ---------------------------
# Script Runner
# ---------------------------
if __name__ == "__main__":
    clf = IntentClassifier()
    clf.train()
