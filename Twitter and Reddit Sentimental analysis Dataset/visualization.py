import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_wordclouds(df):
    """
    Generates Word Clouds for Positive and Negative sentiments.
    """
    print("Generating Word Clouds...")
    
    # Positive Sentiment
    pos_text = " ".join(df[df['category'] == 1]['processed_text'].dropna().astype(str))
    if pos_text:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Positive Sentiment')
        plt.savefig('wordcloud_positive.png')
        plt.close()
        print("Saved wordcloud_positive.png")
    
    # Negative Sentiment
    neg_text = " ".join(df[df['category'] == -1]['processed_text'].dropna().astype(str))
    if neg_text:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(neg_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Negative Sentiment')
        plt.savefig('wordcloud_negative.png')
        plt.close()
        print("Saved wordcloud_negative.png")

def plot_sentiment_distribution(df):
    """
    Creates a Sentiment Distribution bar chart.
    """
    print("Plotting Sentiment Distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='category', data=df, palette='viridis')
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment (1: Pos, 0: Neu, -1: Neg)')
    plt.ylabel('Count')
    plt.savefig('sentiment_distribution.png')
    plt.close()
    print("Saved sentiment_distribution.png")

def plot_time_series(df):
    """
    Creates a Sentiment Trend over time (Time Series) plot.
    Since the dataset lacks a date column, we will simulate dates 
    to demonstrate the visualization capability.
    """
    print("Plotting Time Series (Simulated Dates)...")
    
    # Simulate dates for the last 365 days
    dates = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
    # Shuffle to randomize
    shuffled_dates = np.random.choice(dates, len(df), replace=False)
    df['simulated_date'] = pd.to_datetime(shuffled_dates)
    df['date_only'] = df['simulated_date'].dt.date
    
    # Aggregate sentiment by date (mean sentiment or count of positive)
    daily_sentiment = df.groupby('date_only')['category'].mean()
    
    plt.figure(figsize=(12, 6))
    daily_sentiment.plot(color='purple')
    plt.title('Sentiment Trend Over Time (Simulated)')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.savefig('sentiment_trend.png')
    plt.close()
    print("Saved sentiment_trend.png")

def main():
    data_file = 'processed_data.csv'
    df = load_data(data_file)
    
    if df is not None:
        # Use 'category' or 'textblob_label'
        generate_wordclouds(df)
        plot_sentiment_distribution(df)
        plot_time_series(df)
        print("Visualization complete.")

if __name__ == "__main__":
    main()
