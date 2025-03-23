import streamlit as st
import numpy as np
import pickle
import nltk
import time
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download("vader_lexicon")

# Load the trained model
with open("models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("Emotion Detection from Text")

# User input text box
user_input = st.text_area("Enter a text to analyze:")

# Emotion labels for prediction
emotion_labels = {
    0: "Sadness 😞",
    1: "Joy 😀",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

# Function to change colours in progress bar
def get_progress_bar_color(score):
    if score > 0.2:  
        return "#28a745"  # Green
    elif score < -0.2:  
        return "#dc3545"  # Red
    else:  
        return "#ffc107"  # Yellow


if st.button("Analyze Emotion"):
    if user_input:
        
        # Compute sentiment score
        sentiment_score = sia.polarity_scores(user_input)["compound"]
        
        # Normalize the sentiment score to be in the range 0 to 100
        progress = int(np.clip(abs(sentiment_score) * 100, 0, 100)) 
        
         # Get progress bar color
        progress_bar_color = get_progress_bar_color(sentiment_score)

        # Display the progress bar with smooth animation
        progress_bar = st.empty()
        
        # HTML and CSS for custom progress bar
        progress_html = f"""
        <div style="position: relative; width: 100%; height: 10px; background-color: #ddd; border-radius: 10px; overflow: hidden;">
            <div style="width: 0%; height: 100%; background-color: {progress_bar_color}; transition: width 1s;"></div>
        </div>
        """
        
        # Initially display the empty progress bar
        progress_bar.markdown(progress_html, unsafe_allow_html=True)
        for i in range(1, progress + 1):
            time.sleep(0.02) 
            progress_html = f"""
            <div style="position: relative; width: 100%; height: 10px; background-color: #ddd; border-radius: 10px; overflow: hidden;">
                <div style="width: {i}%; height: 100%; background-color: {progress_bar_color}; transition: width 0.02s;"></div>
            </div>
            """
            progress_bar.markdown(progress_html, unsafe_allow_html=True)

          
        # Transform text using the TF-IDF vectorizer
        input_tfidf = tfidf_vectorizer.transform([user_input])

        # Combine features
        input_features = np.hstack((input_tfidf.toarray(), [[sentiment_score]]))

        # Predict emotion
        prediction = model.predict(input_features)
        predicted_emotion = emotion_labels[prediction[0]]

        # Display result with emoji
        st.subheader(f"Predicted Emotion: {predicted_emotion}")
        st.write(f"Sentiment Score: {sentiment_score:.3f}")
    else:
        st.warning("Please enter some text for analysis!")