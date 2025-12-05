# responses/response_generator.py

import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
from responses.fallback import get_fallback_response
from app.config import CONFIDENCE_THRESHOLD


class ResponseGenerator:
    def __init__(self):
        """
        Load predefined intent-to-response mappings.
        You can later move these to a JSON file if needed.
        """
        self.response_map = {
            "greeting": "Hello! How can I assist you today?",
            "order_status": "Your order status is being checked. Please hold on!",
            "refund_status": "Your refund request is being processed.",
            "contact_support": "You can contact our support team at support@company.com.",
            "goodbye": "Thank you for chatting with us. Have a great day!"
        }

    def generate(self, intent: str, entities: dict, confidence: float) -> str:
        """
        Generate final chatbot response based on:
        - detected intent
        - extracted entities
        - model confidence score
        """

        # 1. Low confidence? Use fallback message
        if confidence < CONFIDENCE_THRESHOLD:
            return get_fallback_response()

        # 2. If intent is known → use mapping
        if intent in self.response_map:
            response = self.response_map[intent]

            # Optional: insert entities into the message
            if entities:
                for key, value in entities.items():
                    placeholder = f"{{{key}}}"
                    if placeholder in response:
                        response = response.replace(placeholder, value)

            return response

        # 3. Unknown intent → fallback
        return get_fallback_response()


# For direct import usage
response_generator = ResponseGenerator()
