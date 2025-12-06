import os
import json
from datetime import datetime


LOG_FILE = "logs/interactions.log"


# Ensure log directory exists
os.makedirs("logs", exist_ok=True)


def log_interaction(user_message, intent, confidence, entities, bot_response):
    """
    Logs every chatbot interaction in a structured JSON line format.

    Parameters:
        user_message (str): Raw user input
        intent (str): Predicted intent label
        confidence (float): Confidence score of intent prediction
        entities (dict): Extracted entities
        bot_response (str): Final generated response
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "bot_response": bot_response
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    except Exception as e:
        print(f"[LOGGER ERROR] Failed to log interaction: {e}")
