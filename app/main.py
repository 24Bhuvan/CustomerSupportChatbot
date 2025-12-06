import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import json

from utils.preprocessing import clean_text
from responses.response_generator import response_generator
from nlu.intent_classifier import IntentClassifier
from nlu.entity_extractor import EntityExtractor
from utils.logger import log_interaction


# ----------------------------
# LOAD MODELS ONCE AT STARTUP
# ----------------------------
@st.cache_resource
def load_intent_model():
    """Load the intent classifier model using joblib."""
    full_path = os.path.abspath("models/intent_model.pkl")
    return joblib.load(full_path)


@st.cache_resource
def load_entity_patterns():
    """Load entity patterns JSON (optional debug)."""
    full_path = os.path.abspath("models/entities.json")
    with open(full_path, "r") as f:
        return json.load(f)


# Load model
intent_model = load_intent_model()
entity_patterns = load_entity_patterns()  # Not used directly, but kept for viewing if needed.

# Configure classifier (attach externally loaded model)
intent_classifier = IntentClassifier(model_path="models/intent_model.pkl")
intent_classifier.model = intent_model

# Entity extractor auto-loads patterns internally
entity_extractor = EntityExtractor()


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Customer Support Chatbot", layout="centered")
st.title("🤖 Customer Support Chatbot")
st.write("Ask any question related to our services!")


# ----------------------------
# CHAT INPUT
# ----------------------------
user_input = st.text_input("Type your message:")

if user_input:

    # 1. Preprocess
    cleaned = clean_text(user_input)

    # 2. Predict intent + confidence
    intent, confidence = intent_classifier.predict_with_confidence(cleaned)

    # 3. Extract entities
    entities = entity_extractor.extract(cleaned)

    # 4. Generate final response
    bot_response = response_generator.generate(intent, entities, confidence)

    # 5. Display on UI
    st.markdown(f"### 🟦 You: {user_input}")
    st.markdown(f"### 🟩 Bot: {bot_response}")

    # 6. Log interaction
    log_interaction(
        user_message=user_input,
        intent=intent,
        confidence=confidence,
        entities=entities,
        bot_response=bot_response
    )
