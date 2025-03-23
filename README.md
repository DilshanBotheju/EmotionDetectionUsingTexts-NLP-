# Emotion Detection Using Texts (NLP)

## Overview
This project is an **NLP-based model** designed to detect emotions in textual data. The system uses **TF-IDF vectorization**, **sentiment analysis**, and **machine learning models** to classify text into different emotion categories.

## Features
- **Preprocessing:** Cleans and processes text data (removal of stopwords, tokenization, etc.).
- **TF-IDF Vectorization:** Converts text into numerical features for model training.
- **Sentiment Analysis:** Uses VADER sentiment analysis for additional feature extraction.
- **Multi-class Classification:** Detects emotions such as **joy, sadness, anger, fear, love, and surprise**.
- **Model Training:** Supports multiple ML models including Logistic Regression and Random Forest.
- **Pickle-based Model Storage:** Saves trained models and vectorizers for future inference.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Emotion-Detection-Using-Texts.git
   cd Emotion-Detection-Using-Texts
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. Save the trained model and vectorizer:
   ```python
   import pickle
   with open('logistic_regression_model.pkl', 'wb') as f:
       pickle.dump(log_reg, f)
   with open('tfidf_vectorizer.pkl', 'wb') as f:
       pickle.dump(tfidf, f)
   ```

## Model Testing
To test the trained model in **Google Colab**:
```python
import pickle

# Load the model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Test input
text = ["I am feeling very happy today!"]
tfidf_text = tfidf_vectorizer.transform(text)
prediction = model.predict(tfidf_text)
print(f"Predicted Emotion: {prediction[0]}")
```

## Dataset
The dataset consists of labeled textual data representing different emotions. Each entry has:
- `cleaned_text`: Processed text data
- `sentiment`: Sentiment score (using VADER)
- `label`: Emotion category (joy, sadness, anger, etc.)

## Models Used
- **Logistic Regression** (Best model, accuracy is 91%)
- **Random Forest** (Caused overfitting and accuracy is 88%)

## Future Improvements
- Implement **transformer-based models** (BERT, RoBERTa) for better accuracy.
- Expand dataset to improve generalization.

## Contributors
- **Dilshan Botheju** - Developer & Researcher
