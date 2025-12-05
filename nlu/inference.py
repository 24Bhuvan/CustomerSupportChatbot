import logging
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nlu.intent_classifier import IntentClassifier
from nlu.entity_extractor import EntityExtractor
from utils.preprocessing import clean_text


# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------
# Load Models Once (Fast)
# ---------------------------
intent_model = IntentClassifier()
entity_model = EntityExtractor()

# Ensure model loads at import time
try:
    intent_model.load()
    logger.info("Intent model loaded for inference.")
except Exception as e:
    logger.error(f"Error loading intent model: {e}")

# Entity extractor auto-loads patterns in __init__()


# ---------------------------
# Main Inference Function
# ---------------------------
def run_inference(user_text: str) -> dict:
    """
    Central pipeline:
    1. Clean text
    2. Predict intent
    3. Extract entities
    4. Return structured JSON output

    Returns:
        {
            "intent": "...",
            "confidence": 0.92,
            "entities": {...}
        }
    """

    if not user_text or user_text.strip() == "":
        return {
            "intent": None,
            "confidence": 0.0,
            "entities": {},
            "error": "Empty input received"
        }

    # 1️⃣ Clean text
    cleaned = clean_text(user_text)

    # 2️⃣ Predict intent + confidence
    try:
        intent, confidence = intent_model.predict_with_confidence(cleaned)
    except Exception as e:
        logger.error(f"Intent prediction error: {e}")
        return {"error": "Model prediction failed"}

    # 3️⃣ Extract entities
    try:
        entities = entity_model.extract(user_text)
    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        entities = {}

    # 4️⃣ Build final response
    result = {
        "intent": intent,
        "confidence": round(confidence, 4),
        "entities": entities
    }

    logger.info(f"Inference Output → {result}")
    return result


# ---------------------------
# Standalone Testing
# ---------------------------
if __name__ == "__main__":
    print("Inference Test Mode\n")

    while True:
        text = input("You: ")

        if text.lower() in ("exit", "quit", "q"):
            break

        output = run_inference(text)
        print("Prediction →", output)
