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
    """
    Production-ready Entity Extractor:
    - Loads rule-based patterns from entities.json
    - Uses spaCy NER for automatic entity detection
    - Merges rule-based + NER results cleanly
    """

    def __init__(self, patterns_path: str = ENTITIES_PATH):
        self.patterns_path = patterns_path
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns: Dict[str, List[str]] = {}

        # Auto-load patterns at initialization
        self._safe_load_patterns()

    # ---------------------------
    # Load Saved Entity Patterns
    # ---------------------------
    def _safe_load_patterns(self):
        """Load patterns safely when the class is created."""
        if not os.path.exists(self.patterns_path):
            logger.warning(
                f"No entity pattern file found at {self.patterns_path}. "
                "Entity extraction will still work with spaCy, but rule-based patterns will be empty."
            )
            self.patterns = {}
            return

        try:
            with open(self.patterns_path, "r") as f:
                self.patterns = json.load(f)
            logger.info("Entity patterns loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading entity patterns: {e}")
            self.patterns = {}

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
        """Extract entities from text using rule-based + spaCy NER."""
        if not text:
            return {}

        entities: Dict[str, str] = {}

        # --- Rule-based Matching ---
        for entity_name, values in self.patterns.items():
            for val in values:
                if val.lower() in text.lower():
                    entities[entity_name] = val

        # --- spaCy Named Entity Recognition ---
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Avoid overwriting rule-based matches
                entities.setdefault(ent.label_, ent.text)
        except Exception as e:
            logger.error(f"spaCy NER error: {e}")

        return entities


# ------------------------------------------------
# Script Runner (for initial pattern creation)
# ------------------------------------------------
if __name__ == "__main__":
    extractor = EntityExtractor()

    # Example patterns — modify later if needed
    sample_patterns = {
        "service": ["analytics", "dashboard", "ai model", "chatbot"],
        "company": ["monarch analytics", "monarch"],
        "contact": ["email", "phone", "support"]
    }

    extractor.save_patterns(sample_patterns)
    logger.info("Sample entity patterns created.")
