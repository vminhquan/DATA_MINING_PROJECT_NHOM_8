from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class FeatureBuilder:
    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))

    def fit_transform_tfidf(self, corpus, save_path=None):
        X = self.vectorizer.fit_transform(corpus)
        if save_path:
            joblib.dump(self.vectorizer, save_path)
        return X

    def transform_tfidf(self, corpus):
        return self.vectorizer.transform(corpus)