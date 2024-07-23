import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to train the model
def train_model(data):
    # Check if 'text' and 'Emotion' columns exist
    if 'text' not in data.columns or 'Emotion' not in data.columns:
        st.error("Error: 'text' or 'Emotion' column not found in the DataFrame.")
        return None, None, None

    # Define features and target
    X = data['text']
    y = data['Emotion']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data with CountVectorizer
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize the Logistic Regression model
    logistic_regression_model = LogisticRegression(max_iter=1000)

    # Train the model
    logistic_regression_model.fit(X_train_vec, y_train)

    # Make predictions
    y_pred = logistic_regression_model.predict(X_test_vec)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Logistic Regression Accuracy: {accuracy:.4f}')
    st.write('Classification Report:')
    st.write(classification_report(y_test, y_pred))

    return logistic_regression_model, vectorizer, accuracy

# Function to predict emotion from text using Logistic Regression
def predict_emotion_logistic(text, model, vectorizer):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

# Title of the web app
st.title('Emotion Prediction using Logistic Regression')

# File upload or file path input
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
file_path = st.text_input("Or enter the file path")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif file_path:
    data = pd.read_csv(file_path)
else:
    data = None

if data is not None:
    st.write("Data Preview:")
    st.write(data.head())

    if st.button('Train Model'):
        model, vectorizer, accuracy = train_model(data)
        if model and vectorizer:
            # Save the model and vectorizer
            joblib.dump(model, 'logistic_regression_model.pkl')
            joblib.dump(vectorizer, 'vectorizer.pkl')
            st.success("Model trained and saved successfully!")

            # Text input for user to enter text
            user_input = st.text_input('Enter text to predict emotion:')

            # Button to make prediction
            if st.button('Predict'):
                if user_input:
                    # Predict emotion
                    predicted_emotion = predict_emotion_logistic(user_input, model, vectorizer)
                    # Display the prediction
                    st.write(f'Predicted Emotion: {predicted_emotion}')
                else:
                    st.write('Please enter some text to predict.')