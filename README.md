# Airline_Review Project
The airline review project aims to analyze and extract from a dataset consisting of reviews of British airlines. Through the application of data science techniques and algorithms, we will explore patterns, sentiment analysis, and other metrics to gain a comprehensive understanding of satisfaction. The project will provide valuable insights to improve airline service, identify areas of improvement, and enhance customer satisfaction

## Table of Contents
Requirements

Installation

Usage

Scraping Airline Reviews

Data Processing

Exploratory Data Analysis (EDA)

Machine Learning Model

Accurary

## Requirements

List the prerequisites and dependencies for your project. For example:

Python 3.7 or higher
Libraries: requests, beautifulsoup4, pandas,numpy,malplotlib,nltk,SentimentIntensityAnalyzer,Labelencoder

## Installation
Provide instructions for installing any necessary dependencies. You can use code blocks to show how to install the required libraries using pip:

pip install requests beautifulsoup4 pandas...etc

## Usage
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from wordcloud import WordCloud

from nltk.sentiment.vader import SentimentIntensityAnalyzer

## Scraping Airline Reviews
 scraping airline reviews from websites. For instance, you can show how to use the requests library and BeautifulSoup to scrape data:

python

import requests

from bs4 import BeautifulSoup

we got the review data from the website  view and after data preprocessing

## Data Preprocessing
It involves the transformation of raw data into a clean and structured format that is suitable for analysis and modeling.This processing typically includes tasks such as removing duplicates, handling missing values, dealing with outliers, normalizing or scaling features, and encoding categorical variables. We can ensure the quality and integrity of the data, improve the accuracy of our models, and obtain meaningful insight from the data.

Cleaning and preprocessing steps
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Remove special characters, numbers, and extra spaces
    
    text = text.lower()
    
    text = ' '.join(word for word in text.split() if word.isalpha())
    
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    return text
    

 df['cleaned_reviews'] = df['reviews'].apply(clean_text)

 ## Exploratory Data Analysis(EDA)
It involves the initial exploration and examination of the dataset to gain insight and understand the underlying patterns, relationships, and characteristics of the data.EDA techniques include summarizing the main statistics of the data, visualization the data using plots and charts identifing and performing statistical tests.


nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df['sentiment_scores'] = df['cleaned_reviews'].apply(lambda x: sia.polarity_scores(x))

df['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x['compound'] >= 0 else 'negative')

# Create a count plot using seaborn

sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))

sns.countplot(data=df, x='sentiment', palette='Set2')  # You can change 'palette' for different color schemes

plt.title('Sentiment Distribution')

plt.xlabel('Sentiment')

plt.ylabel('Count')

 Show the plot

plt.show()

Create Word Clouds
def generate_word_cloud(data, title):

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(data)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

## vader_lexicon
The vader_lexicon is a sentiment analysis tool used for evaluating the sentiment or polarity of text data. It is based on the Valence Aware Dictionary and Sentiment Reasoner (VADER) sentiment analysis tool, which is specifically designed to analyze sentiment in text data, including social media text, product reviews, and more.

Generate word clouds for positive and negative sentiment reviews

positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['cleaned_reviews'])

negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['cleaned_reviews'])

generate_word_cloud(positive_reviews, 'Positive Sentiment Word Cloud')

generate_word_cloud(negative_reviews, 'Negative Sentiment Word Cloud')

## Machine Learning Model

A machine learning model refers to a mathematical algorithm or statistical model that is designed to learn patterns and make predictions or decisions based on input data. it is trained using a dataset that consists of input examples and their corresponding labels
or outcomes. The model learns from the dataset by adjusting its internal parameters to minimize the difference between its predicted output and the true labels.

In this project, we can use Random forest classifier algorithms to predict customer buying behavior

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

### Create a target variable 'Purchase' (1 for customers who purchased, 0 for those who didn't)

### You can define this based on certain criteria, e.g., if they mentioned "booking" or "purchase" in their review text.

df['Purchase'] = df['reviews'].str.contains('booking|purchase', case=False).astype(int)

## Calculate review length

df['review_length'] = df['cleaned_reviews'].apply(len)

## Assuming you have a 'sentiment_scores' column containing sentiment dictionaries

df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])

##Split dataset

X = df[['compound_score', 'review_length']]  # Features

y = df['Purchase']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions

y_pred = clf.predict(X_test)

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')

**Accuracy: 0.86**

## Overview
The accuracy of a machine learning model is a key performance metric that measures the model's ability to make correct predictions on a given dataset. It is an important indicator of how well the model performs its intended task.

Our machine-learning model achieved an accuracy of 0.86 on the test dataset. This means that, out of all the instances in the test dataset, our model correctly predicted 86% of them. In other words, it made correct predictions for 86 out of every 100 instances.




