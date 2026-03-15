from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class RatingRegressor:
    def __init__(self, random_state=42):
        self.models = {
            'Baseline (Ridge)': Ridge(alpha=1.0),
            'Strong Model (Random Forest)': RandomForestRegressor(
                n_estimators=50, random_state=random_state, max_depth=10, n_jobs=-1
            )
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
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = {
                'MAE': mae, 
                'RMSE': rmse, 
                'predictions': y_pred
            }
        return results