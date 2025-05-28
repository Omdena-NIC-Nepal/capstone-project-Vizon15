
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
    analyzer = SentimentIntensityAnalyzer()
    translator = Translator()
    return analyzer, translator

@st.cache_data
def prepare_social_data():
    # Dummy example tokens, in practice load from dataset or user input
    return {'tokens': ['climate', 'change', 'impact', 'weather', 'temperature', 'rainfall', 'flood']}

def run_topic_modeling(tokens, n_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(tokens)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topics = []
    for idx, topic in enumerate(lda.components_):
        words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        topics.append(f"Topic #{idx + 1}: " + ", ".join(words))
    return topics

def run_summarization(text, num_sentences=3):
    doc = english_nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return " ".join(sentences[:num_sentences])
