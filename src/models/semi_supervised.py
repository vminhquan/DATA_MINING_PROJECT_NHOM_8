import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

class SemiSupervisedTrainer:
    def __init__(self, base_model=MultinomialNB()):
        self.semi_model = SelfTrainingClassifier(base_model, threshold=0.8)
        self.sup_model = base_model

    def evaluate_scenario(self, X_train, y_train_true, X_test, y_test, labeled_ratio):
        n_total = X_train.shape[0]
        n_labeled = int(labeled_ratio * n_total)
        
        indices = np.random.permutation(n_total)
        labeled_indices = indices[:n_labeled]
        unlabeled_indices = indices[n_labeled:]
        
        y_train_masked = np.copy(y_train_true)
        y_train_masked[unlabeled_indices] = -1
        
        X_labeled = X_train[labeled_indices]
        y_labeled = y_train_true[labeled_indices]
        self.sup_model.fit(X_labeled, y_labeled)
        y_pred_sup = self.sup_model.predict(X_test)
        f1_sup = f1_score(y_test, y_pred_sup, average='macro')
        
        self.semi_model.fit(X_train, y_train_masked)
        y_pred_semi = self.semi_model.predict(X_test)
        f1_semi = f1_score(y_test, y_pred_semi, average='macro')
        
        return f1_sup, f1_semi, self.semi_model