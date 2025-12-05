import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_intents(input_path, output_path):
    df = pd.read_csv(input_path)

    # clean only the pattern column (others optional)
    df['pattern'] = df['pattern'].apply(clean_text)

    df.to_csv(output_path, index=False)
    print("Preprocessing complete →", output_path)

if __name__ == "__main__":
    preprocess_intents(
        r"nlu/data/intents.csv",
        r"nlu/data/processed_intents.csv"
    )


