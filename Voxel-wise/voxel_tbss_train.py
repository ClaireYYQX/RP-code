import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression


class MLModelTrainer:
    def __init__(self, name, model, X_train, X_test, Y_train, Y_test, Y_group_train, param_grid=None):
        self.name = name
        self.model = model
        self.scaler = StandardScaler()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_train_pred = None
        self.Y_pred = None
        self.Y_group_train = Y_group_train
        self.param_grid = param_grid
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
        # Define the grid search
        age_group = self.Y_group_train
        ckf = self.custom_cv(self.X_train, age_group)
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=ckf, scoring='neg_mean_squared_error', verbose=1)
        print('a')
        grid_search.fit(self.X_train, self.Y_train, groups=age_group)
        print('b')   
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        print(f"Best Parameters: {self.best_params}")
        
    def bias_correction(self):
        self.Y_train_pred = self.best_model.predict(self.X_train)
        # Calculate errors
        errors_train = self.Y_train - self.Y_train_pred
        # Train a linear regression model to predict errors from true ages
        correction_model = LinearRegression()
        correction_model.fit(self.Y_train.reshape(-1, 1), errors_train)
        return correction_model
        
    def predict(self):
        self.Y_pred = self.best_model.predict(self.X_test)
        
        correction_model = self.bias_correction()
        # Use the correction model to estimate errors for the test set predictions
        predicted_errors_test = correction_model.predict(self.Y_pred.reshape(-1, 1))

        # Correct the predictions
        self.Y_pred = self.Y_pred + predicted_errors_test
        
        return self.Y_pred

    def evaluate(self):
        mae = mean_absolute_error(self.Y_test, self.Y_pred)
        r2 = r2_score(self.Y_test, self.Y_pred)
        return mae, r2