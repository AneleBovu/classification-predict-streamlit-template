"""
    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Natural Language Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# Function to perform exploratory data analysis
def perform_eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("EDA allows us to understand the data and its distribution.")

    # Check for missing values
    st.write("Missing values in training data:")
    st.write(data.isnull().sum())

    # Distribution of sentiments
    st.write("Distribution of Sentiments:")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sentiment', data=data)
    plt.title('Distribution of Sentiments')
    st.pyplot()

    # Word cloud
    st.write("Word Cloud:")
    all_words = ' '.join([text for text in data['message']])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot()

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information", "EDA"]  # Added "EDA" option
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']])  # will write the df to the page

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            models = {
                "Logistic Regression": "Logistic_regression.pkl",
                "Random Forest": "rf_model.pkl",
                "Stacked Model": "stacking_model.pkl"
            }
            selected_model = st.selectbox("Select Model", list(models.keys()))
            predictor = joblib.load(open(os.path.join("resources", models[selected_model]), "rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

    # Building out the EDA page
    if selection == "EDA":
        perform_eda(raw)  # Pass the raw data for EDA

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()

