# 💬 Emotion Detector using Machine Learning and Streamlit

This project is a complete end-to-end Machine Learning pipeline that classifies emotions in text using traditional ML techniques. It includes text preprocessing, handling imbalanced data, model training, evaluation, and a deployed web interface using Streamlit.

---

## 📊 Dataset

We used the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset from Hugging Face, which consists of text samples labeled with one of six emotions:

| Label | Emotion   |
|-------|-----------|
| 0     | Sadness   |
| 1     | Joy       |
| 2     | Love      |
| 3     | Anger     |
| 4     | Fear      |
| 5     | Surprise  |

---

## 🧹 Data Preprocessing

### Text Cleaning
The raw text was cleaned using:
- Lowercasing
- Removal of punctuation, numbers, and extra whitespace
- Stopword removal using NLTK
- Lemmatization using `WordNetLemmatizer`

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
````

### Vectorization

Text was converted into numerical form using **TF-IDF Vectorizer**.

---

## ⚖️ Handling Imbalanced Data

We used **SMOTEENN** from `imbalanced-learn` to address class imbalance:

* **SMOTE**: Oversamples minority classes
* **ENN**: Cleans noisy examples from majority classes

```python
from imblearn.combine import SMOTEENN
```

---

## 🤖 Model Training

We trained a **Logistic Regression** model with the **One-vs-Rest** strategy using Scikit-learn:

* `LogisticRegression(max_iter=1000, class_weight='balanced')`
* Trained on the SMOTEENN-balanced dataset
* Used `LabelEncoder` for encoding emotion labels

---

## 📈 Evaluation

### Metrics Used:

* Accuracy
* F1-score
* Precision & Recall
* Confusion Matrix

### Sample Results:

| Set        | Accuracy |
| ---------- | -------- |
| Validation | \~85.75% |
| Test       | \~84.55% |

### Common Confusions:

* **Joy ↔ Love**
* **Sadness ↔ Fear**

---

## 💾 Model Saving

We saved all important components using `joblib`:

* `emotion_model.pkl`: Trained One-vs-Rest classifier
* `tfidf_vectorizer.pkl`: TF-IDF vectorizer
* `label_encoder.pkl`: Encoded emotion labels

---

## 🌐 Streamlit Web App

We built an interactive front-end using **Streamlit**, allowing users to enter text and get real-time emotion predictions.

### Features:

* Clean and user-friendly UI
* Real-time emotion detection
* Themed UI (custom color themes)
* Deployed online via Streamlit Cloud

### Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧪 Sample Predictions

| Input Text                                     | Predicted Emotion |
| ---------------------------------------------- | ----------------- |
| "I feel so blessed and grateful today!"        | Joy               |
| "I'm scared to take the next step in my life." | Fear              |
| "I miss you so much it hurts."                 | Sadness           |
| "You lied to me again, I can't believe it!"    | Anger             |
| "Wow! I wasn't expecting that!"                | Surprise          |

---

## 📁 Project Structure

```
emotiondetector/
├── main.ipynb               
├── emotion_model.pkl 
├── app.py                   # Streamlit app
├── emotion_model.pkl        # Trained model
├── tfidf_vectorizer.pkl     # Vectorizer
├── label_encoder.pkl        # Label encoder
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Custom theme settings

```

---

## 🚀 Deployment

Hosted on **Streamlit Cloud**:
* [Link of deployment](https://emotiondetector-yc48mhyes3ywy9ytpxs3sa.streamlit.app/)
---

## 🤝 Acknowledgements

* [Hugging Face Datasets](https://huggingface.co/datasets/dair-ai/emotion)
* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)
* [NLTK](https://www.nltk.org/)
* [imbalanced-learn](https://imbalanced-learn.org/)

---


