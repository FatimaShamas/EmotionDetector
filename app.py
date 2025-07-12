import streamlit as st
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Load components
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🌸 Emotion Classifier 🌸")
st.write("Enter a sentence to predict the underlying emotion.")

user_input = st.text_area("Text Input", placeholder="e.g. I feel anxious today!")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        X_input = vectorizer.transform([cleaned])
        pred = model.predict(X_input)
        emotion = label_encoder.inverse_transform(pred)[0]

        st.success(f"Predicted Emotion: **{label_names[emotion].capitalize()}**")

        # Optional: Show all class probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            st.subheader("Class Probabilities")
            for i, prob in enumerate(probs):
                st.write(f"{label_names[i]}: {prob:.2f}")
