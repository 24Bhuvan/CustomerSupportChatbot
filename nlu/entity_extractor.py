import os
import json
import logging
from typing import Dict, List

import spacy


# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

ENTITIES_PATH = "models/entities.json"


class EntityExtractor:
    def __init__(self, patterns_path: str = ENTITIES_PATH):
        self.patterns_path = patterns_path
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = {}

    # ---------------------------
    # Load Saved Entity Patterns
    # ---------------------------
    def load_patterns(self):
        if not os.path.exists(self.patterns_path):
            raise FileNotFoundError("Entity patterns not found. Create them first.")

        with open(self.patterns_path, "r") as f:
            self.patterns = json.load(f)

        logger.info("Entity patterns loaded.")

    # ---------------------------
    # Save Entity Patterns
    # ---------------------------
    def save_patterns(self, patterns: Dict[str, List[str]]):
        os.makedirs("models", exist_ok=True)

        with open(self.patterns_path, "w") as f:
            json.dump(patterns, f, indent=4)

        self.patterns = patterns
        logger.info(f"Entity patterns saved at {self.patterns_path}")

    # ---------------------------
    # Extract Entities (Rule-based + spaCy NER)
    # ---------------------------
    def extract(self, text: str) -> Dict[str, str]:
        entities = {}

        # --- Rule-based Matching ---
        for entity_name, values in self.patterns.items():
            for val in values:
                if val.lower() in text.lower():
                    entities[entity_name] = val

        # --- spaCy Named Entity Recognition ---
        doc = self.nlp(text)
        for ent in doc.ents:
            entities[ent.label_] = ent.text

        return entities


# ------------------------------------------------
# Script Runner (for creating initial rule patterns)
# ------------------------------------------------
if __name__ == "__main__":
    extractor = EntityExtractor()

    # Example entity patterns — you can modify this later
    sample_patterns = {
        "service": ["analytics", "dashboard", "ai model", "chatbot"],
        "company": ["monarch analytics", "monarch"],
        "contact": ["email", "phone", "support"]
    }

    extractor.save_patterns(sample_patterns)
    logger.info("Sample entity patterns created.")
