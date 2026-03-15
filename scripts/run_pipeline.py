import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.cleaner import TextCleaner
from src.features.builder import FeatureBuilder
from src.models.supervised import SentimentClassifier
from src.models.regression import RatingRegressor

def get_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

def main():
    print("Starting Data Mining Pipeline...")
    
    df = pd.read_csv('data/raw/7282_1.csv')
    df = df.dropna(subset=['reviews.rating', 'reviews.text'])
    df = df.drop_duplicates(subset=['reviews.text']).reset_index(drop=True)
    
    cleaner = TextCleaner()
    df['cleaned_text'] = df['reviews.text'].apply(cleaner.clean_text)
    df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    df['sentiment'] = df['reviews.rating'].apply(get_sentiment)
    df.to_csv('data/processed/cleaned_reviews.csv', index=False)
    
    print("Data cleaning completed. Building features...")
    
    builder = FeatureBuilder(max_features=3000)
    tfidf_matrix = builder.fit_transform_tfidf(df['cleaned_text'], save_path='outputs/models/tfidf_vectorizer.pkl')
    sparse.save_npz('data/processed/tfidf_matrix.npz', tfidf_matrix)
    
    print("Features built. Training Classification Models...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    clf = SentimentClassifier(random_state=42)
    clf.train_all(X_train, y_train)
    clf_results = clf.evaluate_all(X_test, y_test)
    
    for name, res in clf_results.items():
        print(f"Classification | {name} - Macro F1: {res['macro_f1']:.4f}")
        
    print("Training Regression Models...")
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        tfidf_matrix, df['reviews.rating'], test_size=0.2, random_state=42
    )
    
    reg = RatingRegressor(random_state=42)
    reg.train_all(X_train_reg, y_train_reg)
    reg_results = reg.evaluate_all(X_test_reg, y_test_reg)
    
    for name, res in reg_results.items():
        print(f"Regression | {name} - MAE: {res['MAE']:.4f}, RMSE: {res['RMSE']:.4f}")

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()