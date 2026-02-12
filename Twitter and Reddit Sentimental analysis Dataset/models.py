import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import gc
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Loads the processed dataset with optimized types."""
    try:
        # Load only necessary columns
        df = pd.read_csv(filepath, usecols=['processed_text', 'category'])
        df.dropna(subset=['processed_text', 'category'], inplace=True)
        
        # Optimize types
        df['category'] = df['category'].astype('int8')
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_tfidf_logreg(X_train, X_test, y_train, y_test):
    """
    Trains a Logistic Regression model with TF-IDF vectorization.
    """
    print("\n--- Training TF-IDF + Logistic Regression ---")
    
    # Vectorization
    # Limit features to 5000 to save memory
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), dtype=np.float32)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Garbage collect raw text to free memory
    gc.collect()
    
    # Model
    # Updated: removed multi_class, used solver='lbfgs' and max_iter=1000 as requested
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluation
    metrics = evaluate_model(y_test, y_pred, "Logistic Regression")
    
    return model, vectorizer, metrics

def simple_deep_learning_model(X_train, X_test, y_train, y_test, num_classes=3):
    """
    Implements a simplified Deep Learning model (2-Layer Dense).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.utils import to_categorical
        from sklearn.preprocessing import LabelEncoder
        
        print("\n--- Training Deep Learning Model (Simplified Dense) ---")

        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
        y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
        
        # TF-IDF Vectorization for DL
        # Use float32 to save memory
        vectorizer = TfidfVectorizer(max_features=4000, dtype=np.float32)
        X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = vectorizer.transform(X_test).toarray()
        
        # Clear memory
        gc.collect()
        
        # Simplified Architecture
        model = Sequential([
            Dense(128, input_shape=(4000,), activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Train
        model.fit(X_train_tfidf, y_train_cat, epochs=5, batch_size=64, validation_split=0.1, verbose=0)
        
        # Evaluate
        y_pred_prob = model.predict(X_test_tfidf)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Inverse transform labels
        y_pred_labels = le.inverse_transform(y_pred)
        
        metrics = evaluate_model(y_test, y_pred_labels, "Deep Learning")
        
        # Function-level cleanup
        del X_train_tfidf, X_test_tfidf, model
        gc.collect()
        
        return metrics
        
    except ImportError:
        print("TensorFlow/Keras not available. Skipping DL model.")
        return {}
    except Exception as e:
        print(f"Error in DL model: {e}")
        return {}

def evaluate_model(y_true, y_pred, model_name):
    """
    Calculates metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

def print_comparison_table(lr_metrics, dl_metrics):
    """
    Prints a clear comparison table.
    """
    print("\n" + "="*65)
    print(f"{'Metric':<20} | {'Logistic Regression':<20} | {'Deep Learning':<20}")
    print("-" * 65)
    
    metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for key in metrics_keys:
        lr_val = lr_metrics.get(key, 0)
        dl_val = dl_metrics.get(key, 0) if dl_metrics else 0
        print(f"{key:<20} | {lr_val:.4f}{' '*14} | {dl_val:.4f}")
        
    print("="*65 + "\n")

    # Winner logic
    lr_acc = lr_metrics.get('Accuracy', 0)
    dl_acc = dl_metrics.get('Accuracy', 0) if dl_metrics else 0
    
    if dl_acc > lr_acc:
        print(f"🏆 Deep Learning performed better by {(dl_acc - lr_acc)*100:.2f}%")
    else:
        print(f"🏆 Logistic Regression performed better by {(lr_acc - dl_acc)*100:.2f}%")

def main():
    data_file = 'processed_data.csv'
    df = load_data(data_file)
    
    if df is not None:
        X = df['processed_text']
        y = df['category']
        
        gc.collect()
        
        print("Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. TF-IDF + Logistic Regression
        lr_model, vectorizer, lr_metrics = train_tfidf_logreg(X_train, X_test, y_train, y_test)
        
        del lr_model, vectorizer
        gc.collect()
        
        # 2. Simplified Deep Learning
        dl_metrics = simple_deep_learning_model(X_train, X_test, y_train, y_test)
        
        # 3. Model Comparison Table
        print_comparison_table(lr_metrics, dl_metrics)

if __name__ == "__main__":
    main()
