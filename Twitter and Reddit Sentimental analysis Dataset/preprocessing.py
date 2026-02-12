import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import gc

# Download NLTK data
# Added 'punkt_tab' as requested to avoid resource errors
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

def load_and_merge_data(reddit_path, twitter_path):
    """
    Loads, merges, and samples the dataset to optimize memory usage.
    """
    try:
        print("Loading datasets...")
        # Load Reddit Data
        df_reddit = pd.read_csv(reddit_path)
        df_reddit = df_reddit.rename(columns={'clean_comment': 'text'})
        df_reddit['source'] = 'Reddit'
        
        # Load Twitter Data
        df_twitter = pd.read_csv(twitter_path)
        df_twitter = df_twitter.rename(columns={'clean_text': 'text'})
        df_twitter['source'] = 'Twitter'
        
        # Merge
        df = pd.concat([df_reddit, df_twitter], ignore_index=True)
        
        # Drop rows with missing text
        df.dropna(subset=['text'], inplace=True)
        
        print(f"Total records before sampling: {len(df)}")
        
        # Optimize 1: Data Sampling (25,000 - 30,000 records)
        target_sample_size = 30000
        if len(df) > target_sample_size:
            print(f"Sampling down to {target_sample_size} records for performance...")
            df = df.sample(n=target_sample_size, random_state=42)
        
        # Memory cleanup
        del df_reddit, df_twitter
        gc.collect()
        
        # Handle labels: ensure category is integer and valid
        # Casting to int8 to save memory
        df['category'] = pd.to_numeric(df['category'], errors='coerce')
        df.dropna(subset=['category'], inplace=True)
        df['category'] = df['category'].astype('int8')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_text(text):
    """
    Cleans text by removing URLs, emojis, special characters.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove Emojis/Special chars (keep only alphabets and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text_batch(texts, lemmatizer, stop_words):
    """
    Batch processing for tokenization and lemmatization.
    """
    processed_texts = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if i % 5000 == 0:
            print(f"Processed {i}/{total} records...")
            
        if not text:
            processed_texts.append("")
            continue
            
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        cleaned_tokens = [
            lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in stop_words and len(token) > 2
        ]
        processed_texts.append(" ".join(cleaned_tokens))
    
    return processed_texts

def get_textblob_sentiment(text):
    """
    Returns sentiment label: 1 (Positive), -1 (Negative), 0 (Neutral).
    Type: int8
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity < 0:
        return -1
    else:
        return 0

def main():
    reddit_file = 'Reddit_Data.csv'
    twitter_file = 'Twitter_Data.csv'
    
    df = load_and_merge_data(reddit_file, twitter_file)
    
    if df is not None:
        print("Cleaning text...")
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Free up memory from raw text column if not needed, 
        # but we might keep it for debug. Let's optimize if needed.
        # df.drop(columns=['text'], inplace=True) 
        gc.collect()
        
        print("Tokenizing and Lemmatizing...")
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Processing
        df['processed_text'] = preprocess_text_batch(df['cleaned_text'], lemmatizer, stop_words)
        
        # Optimize 2: Clear memory after heavy NLP tasks
        gc.collect()
        
        # Remove empty
        df = df[df['processed_text'] != ""]
        
        print("Applying TextBlob Sentiment Labeling...")
        # Return result as int8
        df['textblob_label'] = df['processed_text'].apply(get_textblob_sentiment).astype('int8')
        
        output_file = 'processed_data.csv'
        print(f"Saving to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Preprocessing complete.")
        
        # Final cleanup
        del df
        gc.collect()
        
    else:
        print("Failed to process data.")

if __name__ == "__main__":
    main()
