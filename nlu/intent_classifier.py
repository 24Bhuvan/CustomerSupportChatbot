import os
import joblib
import logging
from typing import Tuple, Optional

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
    """
    Production-ready Intent Classifier:
    - Trains a TF-IDF + Logistic Regression model
    - Saves the trained model
    - Loads model automatically for inference
    - Predicts intent + confidence scores
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[Pipeline] = None

    # ---------------------------
    # Load Dataset
    # ---------------------------
    def load_data(self, path: str = DATA_PATH) -> Tuple[pd.Series, pd.Series]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        if "pattern" not in df.columns or "intent" not in df.columns:
            raise KeyError("CSV must contain 'pattern' (text) and 'intent' columns.")

        return df["pattern"], df["intent"]

    # ---------------------------
    # Train Model and Save
    # ---------------------------
    def train(self):
        logger.info("Loading training dataset...")
        X, y = self.load_data()

        logger.info("Splitting dataset into train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Building ML pipeline (TF-IDF + Logistic Regression)...")
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=300))
        ])

        logger.info("Training intent classifier...")
        pipeline.fit(X_train, y_train)

        logger.info("Evaluating model...")
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + classification_report(y_test, predictions))

        logger.info("Saving trained model...")
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, self.model_path)

        logger.info(f"Model saved at {self.model_path}")

    # ---------------------------
    # Load Saved Model
    # ---------------------------
    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Train the model first."
            )
        self.model = joblib.load(self.model_path)
        logger.info("Intent model loaded successfully.")

    # ---------------------------
    # Predict Intent Only
    # ---------------------------
    def predict(self, text: str) -> str:
        if self.model is None:
            self.load()
        return self.model.predict([text])[0]

    # ---------------------------
    # Predict Intent + Confidence
    # ---------------------------
    def predict_with_confidence(self, text: str):
        """
        Returns:
            intent: predicted label
            confidence: float (0–1)
        """
        if self.model is None:
            self.load()

        # Probability scores for each class
        probabilities = self.model.predict_proba([text])[0]
        intent = self.model.predict([text])[0]
        confidence = max(probabilities)

        return intent, float(confidence)


# ---------------------------
# Script Runner for Training
# ---------------------------
if __name__ == "__main__":
    classifier = IntentClassifier()
    classifier.train()
