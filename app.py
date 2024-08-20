import os
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    client = MongoClient(os.getenv('MONGODB_URI'))
    db = client[os.getenv('MONGODB_DB')]
    collection = db[os.getenv('MONGODB_COLLECTION')]
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {e}")
    raise

try:
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    aspect_model = AutoModelForSeq2SeqLM.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    aspect_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
    aspect_pipeline = pipeline("text2text-generation", model=aspect_model, tokenizer=aspect_tokenizer)
    
    logger.info("Successfully loaded NLP models")
except Exception as e:
    logger.error(f"Error loading NLP models: {e}")
    raise

def preprocess_text(text: str) -> List[str]:
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.isalpha() and token not in stop_words]

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    tokens = preprocess_text(text)
    return [word for word, _ in Counter(tokens).most_common(top_n)]

def perform_aspect_based_sentiment_analysis(text: str) -> Dict[str, float]:
    result = aspect_pipeline(text, max_length=512, clean_up_tokenization_spaces=True)
    aspects = result[0]['generated_text'].split(', ')
    aspect_sentiments = {}
    for aspect in aspects:
        if ':' in aspect:
            aspect_name, sentiment = aspect.split(':')
            aspect_sentiments[aspect_name.strip()] = float(sentiment)
    return aspect_sentiments

def extract_named_entities(text: str) -> List[str]:
    blob = TextBlob(text)
    return [item for item in blob.noun_phrases if len(item.split()) > 1]

def perform_topic_modeling(texts: List[str], num_topics: int = 5, num_words: int = 10) -> List[List[str]]:
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    feature_names = vectorizer.get_feature_names()
    return [[feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]] for topic in lda.components_]

def analyze_feedback(feedback: str) -> Dict[str, Any]:
    try:
        sentiment = sentiment_pipeline(feedback)[0]
        topics = ["product quality", "customer service", "pricing", "user experience", "feature request", "bug report"]
        topic_result = zero_shot_classifier(feedback, topics)
        keywords = extract_keywords(feedback)
        named_entities = extract_named_entities(feedback)
        aspect_sentiments = perform_aspect_based_sentiment_analysis(feedback)
        summary = summarizer(feedback, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        
        return {
            "feedback": feedback,
            "sentiment": sentiment['label'],
            "sentiment_score": sentiment['score'],
            "topic": topic_result['labels'][0],
            "topic_score": topic_result['scores'][0],
            "keywords": keywords,
            "named_entities": named_entities,
            "aspect_sentiments": aspect_sentiments,
            "summary": summary,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error in analyze_feedback: {e}")
        raise

def store_feedback(analyzed_feedback: Dict[str, Any]) -> None:
    try:
        collection.insert_one(analyzed_feedback)
        logger.info(f"Stored feedback: {analyzed_feedback['feedback'][:50]}...")
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise

def get_all_feedback() -> List[Dict[str, Any]]:
    try:
        return list(collection.find())
    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        raise

def create_dashboard():
    st.set_page_config(page_title="Advanced Customer Feedback Analysis", layout="wide")
    st.title("Advanced Customer Feedback Analysis Dashboard")

    all_feedback = get_all_feedback()
    df = pd.DataFrame(all_feedback)

    st.sidebar.header("Filters")
    date_range = st.sidebar.date_input("Select Date Range", [df['timestamp'].min(), df['timestamp'].max()])
    selected_topics = st.sidebar.multiselect("Select Topics", df['topic'].unique())
    
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    if selected_topics:
        mask &= df['topic'].isin(selected_topics)
    df_filtered = df.loc[mask]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        fig_sentiment = px.pie(df_filtered, names='sentiment', title='Sentiment Distribution')
        st.plotly_chart(fig_sentiment, use_container_width=True)

    with col2:
        st.subheader("Topic Distribution")
        fig_topic = px.bar(df_filtered['topic'].value_counts().reset_index(), x='index', y='topic', title='Topic Distribution')
        st.plotly_chart(fig_topic, use_container_width=True)

    st.subheader("Sentiment Over Time")
    df_filtered['date'] = df_filtered['timestamp'].dt.date
    sentiment_over_time = df_filtered.groupby('date')['sentiment_score'].mean().reset_index()
    fig_sentiment_time = px.line(sentiment_over_time, x='date', y='sentiment_score', title='Average Sentiment Score Over Time')
    st.plotly_chart(fig_sentiment_time, use_container_width=True)

    st.subheader("Keyword Cloud")
    all_keywords = [keyword for keywords in df_filtered['keywords'] for keyword in keywords]
    keyword_freq = Counter(all_keywords)
    fig_wordcloud = go.Figure(data=[go.Bar(x=list(keyword_freq.keys()), y=list(keyword_freq.values()))])
    fig_wordcloud.update_layout(title='Top Keywords', xaxis_title='Keyword', yaxis_title='Frequency')
    st.plotly_chart(fig_wordcloud, use_container_width=True)

    st.subheader("Recent Feedback")
    for _, row in df_filtered.sort_values('timestamp', ascending=False).head(5).iterrows():
        with st.expander(f"{row['sentiment']} - {row['topic']} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            st.write(f"Feedback: {row['feedback']}")
            st.write(f"Summary: {row['summary']}")
            st.write(f"Keywords: {', '.join(row['keywords'])}")
            st.write(f"Named Entities: {', '.join(row['named_entities'])}")
            st.write("Aspect Sentiments:")
            for aspect, sentiment in row['aspect_sentiments'].items():
                st.write(f"- {aspect}: {sentiment}")

    st.subheader("Analyze New Feedback")
    new_feedback = st.text_area("Enter feedback here:")
    if st.button("Analyze"):
        if new_feedback:
            with st.spinner("Analyzing feedback..."):
                analysis = analyze_feedback(new_feedback)
                store_feedback(analysis)
                st.json(analysis)
        else:
            st.warning("Please enter some feedback to analyze.")

    if st.button("Perform Topic Modeling"):
        with st.spinner("Performing topic modeling..."):
            topics = perform_topic_modeling(df_filtered['feedback'].tolist())
            st.subheader("Discovered Topics")
            for i, topic in enumerate(topics):
                st.write(f"Topic {i+1}: {', '.join(topic)}")

if __name__ == "__main__":
    create_dashboard()
