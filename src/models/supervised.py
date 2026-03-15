from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

class SentimentClassifier:
    def __init__(self, random_state=42):
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Linear SVC': LinearSVC(random_state=random_state, dual=False)
        }
        self.trained_models = {}

    def train_all(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate_all(self, X_test, y_test):
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            results[name] = {
                'macro_f1': macro_f1,
                'report': classification_report(y_test, y_pred),
                'matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
        return results