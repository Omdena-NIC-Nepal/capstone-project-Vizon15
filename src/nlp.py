import os
# Prevent parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
from datetime import datetime, timedelta

import spacy

# Load spaCy's English model
english_nlp = spacy.load("en_core_web_sm")

# Get English stopwords from spaCy
english_stopwords = english_nlp.Defaults.stop_words
# Minimal in-memory stopwords list (subset)
STOPWORDS = set([
    'the','and','is','in','to','of','for','on','with','a','an','that','this','as','are','be','by','it','from'
])

# Cache lightweight sentiment analyzer and translator
@st.cache_resource
def load_nlp_tools():
    """
    Loads tools that are light on memory:
     - VADER SentimentIntensityAnalyzer
     - Google Translator
    """
    sentiment_analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    return {'sentiment': sentiment_analyzer, 'translator': translator}

# Fetch news articles
@st.cache_data
def get_news_articles():
    urls = [
        'https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature',
        'https://www.un.org/en/climatechange/news'
    ]
    articles = []
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            articles.append({'url': url, 'text': text})
        except:
            continue
    return pd.DataFrame(articles)

# Preprocess text: lowercase, simple tokenization, remove stopwords
def preprocess_text(text):
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]

# Sentiment Analysis using VADER
def analyze_sentiment(text):
    tools = load_nlp_tools()
    return tools['sentiment'].polarity_scores(text)

# Simple Named Entity Recognition: extract capitalized sequences
def extract_entities(text):
    # Regex to find capitalized words or phrases
    entities = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    # Filter out stopwords
    return [e for e in entities if e.lower() not in STOPWORDS]

# Topic Modeling with LDA (sklearn)
stopwords_list = list(STOPWORDS)
vect = CountVectorizer(max_features=5000, stop_words=stopwords_list)

def run_topic_modeling(texts, n_topics=5, n_words=10):
    """
    Perform topic modeling on a list of texts using LDA.
    
    Args:
        texts (list of str): List of input texts.
        n_topics (int): Number of topics to extract.
        n_words (int): Number of top words to display per topic.
    
    Returns:
        dict: A dictionary where keys are topic numbers and values are lists of top words.
    """
    # Vectorize the text data
    dtm = vect.fit_transform(texts)
    
    # Fit the LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    
    # Extract topics and their top words
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vect.get_feature_names_out()[i] for i in topic.argsort()[-n_words:][::-1]]
        topics[f"Topic {topic_idx + 1}"] = top_words
    
    return topics

# Text Summarization: extract first N sentences
def run_summarization(text, n_sentences=3):
    """
    Summarizes the given text into the specified number of sentences.

    Args:
        text (str): The input text to summarize.
        n_sentences (int): The number of sentences to include in the summary.

    Returns:
        str: The summarized text containing the specified number of sentences.
    """
    # Split the text into sentences using regex
    sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    
    # Handle edge cases where the text has fewer sentences than requested
    if len(sents) == 0:
        return ""  # Return an empty string if no sentences are found
    elif len(sents) < n_sentences:
        n_sentences = len(sents)  # Adjust n_sentences to the available number of sentences
    
    # Join the first `n_sentences` sentences into the summary
    summary = ' '.join(sents[:n_sentences])
    
    # Ensure the summary ends with proper punctuation
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary

# Translation using Googletrans
def translate(text, src='auto', tgt='en'):
    tools = load_nlp_tools()
    try:
        return tools['translator'].translate(text, src=src, dest=tgt).text
    except:
        return text

# Prepare synthetic social data
@st.cache_data
def prepare_social_data(n=100):
    sample = [
        "Heatwave hits Kathmandu Valley, temperatures soar past 40°C.",
        "Flash floods devastate villages in Karnali Province.",
        "Reforestation efforts in Chitwan show promising results.",
        "Air quality in Pokhara deteriorates due to wildfires.",
        "Monsoon patterns shifting, affecting agriculture in Terai region.",
        "Glacial melt accelerates in the Himalayas, raising sea levels.",
        "Community-led clean-up drives gain momentum in Lalitpur.",
        "Solar energy adoption increases in rural Nepal.",
        "Drought conditions worsen in western districts.",
        "Climate change education programs launched in schools."
    ]
    hashtags = ["#ClimateChange", "#Nepal", "#Environment", "#Sustainability", "#GlobalWarming"]
    locations = ["Kathmandu", "Pokhara", "Lalitpur", "Biratnagar", "Chitwan", "Dharan", "Butwal", "Nepalgunj"]
    
    data = []
    for _ in range(n):
        text = random.choice(sample)
        if random.random() > 0.5:
            text += f" {random.choice(hashtags)}"
        if random.random() > 0.7:
            text += f" in {random.choice(locations)}"
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))
        data.append({"date": timestamp, "text": text})
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' column is datetime type

    df['tokens'] = df['text'].apply(preprocess_text)
    df['sentiment'] = df['text'].apply(analyze_sentiment)
    df['entities'] = df['text'].apply(extract_entities)
    df['summary'] = df['text'].apply(lambda x: run_summarization(x, 2))
    return df

# Integrate NLP insights with climate data
@st.cache_data
def integrate_nlp_with_climate(social_df, climate_csv):
    clim = pd.read_csv(climate_csv, parse_dates=['Date'])
    clim['Date'] = clim['Date'].dt.normalize()
    social_df['date'] = pd.to_datetime(social_df.get('date', pd.Timestamp.now())).normalize()
    sent = social_df.groupby('date')['sentiment'].apply(lambda lst: np.mean([d['compound'] for d in lst]))
    sent = sent.reset_index(name='avg_sentiment')
    merged = clim.merge(sent, left_on='Date', right_on='date', how='left')
    return merged.drop(columns=['date'])

# Expose tools
TOOLS = load_nlp_tools()