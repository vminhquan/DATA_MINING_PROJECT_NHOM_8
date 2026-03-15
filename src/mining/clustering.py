from sklearn.cluster import KMeans
import pandas as pd

class TopicClustering:
    def __init__(self, n_clusters=5, random_state=42):
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit_predict(self, tfidf_matrix):
        return self.model.fit_predict(tfidf_matrix)

    def get_top_terms_per_cluster(self, vectorizer, n_terms=10):
        terms = vectorizer.get_feature_names_out()
        top_terms = {}
        for i in range(self.model.n_clusters):
            cluster_center = self.model.cluster_centers_[i]
            sorted_terms = cluster_center.argsort()[::-1][:n_terms]
            top_terms[f"Cluster {i}"] = [terms[idx] for idx in sorted_terms]
        return top_terms