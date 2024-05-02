import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class MLModelTrainer:
    def __init__(self, name, model, x_train, x_test, y_train, y_test, y_group_train, param_grid=None):
        self.name = name
        self.model = model
        self.scaler = StandardScaler()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_pred = None
        self.y_pred = None
        self.y_group_train = y_group_train
        self.param_grid = param_grid
        self.cv_folds = 5
        self.best_model = None
        self.best_params = None
        self.best_features = None
        self.mean_score = None

    def custom_cv(self, x, y_bins, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits)
        for train_idx, test_idx in skf.split(x, y_bins):
            print(f"Train fold distribution: \n{y_bins.iloc[train_idx].value_counts(normalize=True)}")
            print(f"Test fold distribution: \n{y_bins.iloc[test_idx].value_counts(normalize=True)}")
            yield train_idx, test_idx
        
    def train(self):
        
        print(self.model)
 
        age_group = self.y_group_train
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=self.custom_cv(self.x_train, age_group), scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(self.x_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best Parameters: {self.best_params}")


    def bias_correction(self):
        self.y_train_pred = self.best_model.predict(self.x_train)
        # Calculate errors
        errors_train = self.y_train - self.y_train_pred
        # Train a linear regression model to predict errors from true ages
        correction_model = LinearRegression()
        correction_model.fit(self.y_train.values.reshape(-1, 1), errors_train)
        return correction_model
        
    def predict(self):
        self.y_pred = self.best_model.predict(self.x_test)
        
        correction_model = self.bias_correction()
        # Use the correction model to estimate errors for the test set predictions
        predicted_errors_test = correction_model.predict(self.y_pred.reshape(-1, 1))

        # Correct the predictions
        self.y_pred = self.y_pred + predicted_errors_test
        
        return self.y_pred

    def evaluate(self):
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        return mae, r2