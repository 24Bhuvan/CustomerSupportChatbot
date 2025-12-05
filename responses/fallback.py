# responses/fallback.py

def get_fallback_response() -> str:
    """
    Return a safe default message when the model's confidence is low
    or when an unknown intent is detected.
    """
    return (
        "I'm sorry, I didn't quite understand that. "
        "Could you please rephrase your question?"
    )
