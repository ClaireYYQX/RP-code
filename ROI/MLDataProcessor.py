import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer

class MLDataProcessor:
    def __init__(self, modality, ROI_method):
        self.df = None
        self.modality = modality
        self.ROI_method = ROI_method

    def ROI_df(self):
        self.df = pd.read_csv(f'data/ROI_{self.ROI_method}_{self.modality}_df_4IQR.csv').drop('Unnamed: 0', axis=1)
        # self.df = pd.read_csv(f'{self.modality}_subsample.csv')

    def stratified_sampling(self, random_seed):
        target_column = 'age'
        # Define age groups
        age_bins = [18, 20, 30, 40, 50, 60, 70, 80, 90]
        age_labels = ['18-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80','80-90']
        # age_bins = [20, 30, 40, 50, 60, 70, 85]
        # age_labels = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-85']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels, right=False)
        x = self.df.drop([target_column, 'age_group'], axis=1)
        y = self.df['age']
        y_group = self.df['age_group']
        counts = self.df.groupby('age_group').size()
        print(counts)
        
        # use this if need mutiple splits, i.e. need in CV
        folds_data = []
        stratify_values = self.df['age_group'].values.ravel()
        splitter = StratifiedKFold(n_splits=10, random_state=random_seed, shuffle=True)
        for train_index, test_index in splitter.split(x, stratify_values):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            y_group_train = y_group.iloc[train_index]
            # print(x_test)
            fold_data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test, 'y_group_train': y_group_train}
            folds_data.append(fold_data)
            print(y_test)

        return folds_data
    
    def z_score_normalisation(self, x_train, x_test):
        columns=x_train.columns
        
        # Creating a StandardScaler object
        scaler = StandardScaler()

        # Fitting and transforming the training data
        x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=columns)

        #  Only transforming the test data
        x_test = pd.DataFrame(scaler.transform(x_test), columns=columns)
        
        columns=x_train.columns

        return x_train, x_test
    
    def Factor_Analysis(self, X_train, X_test):
        # Perform factor analysis on the training data
        fa = FactorAnalyzer(rotation="varimax", n_factors=5)
        fa.fit(X_train)
    
        # Determine the optimal number of factors
        ev, v = fa.get_eigenvalues()
        n_factors = sum(ev > 1)  # Number of factors to retain
        print(n_factors)
        fa = FactorAnalyzer(rotation="varimax", n_factors=n_factors)
        fa.fit(X_train)
    
        # Transform both training and test datasets
        X_train_transformed = fa.transform(X_train)
        X_test_transformed = fa.transform(X_test)

        # Get variance explained by each factor
        variance = fa.get_factor_variance()
        print("Variance Explained (each factor):", variance[1])
        print("Cumulative Variance Explained:", variance[2])
        
        weighted_scores = {}
        
        for i, feature in enumerate(fa.loadings_, start=1):
            weighted_scores[X_train.columns[i-1]] = np.dot(abs(feature), variance[1])
        
        # Calculate weighted score for each feature
        
        # print(weighted_scores)
        weighted_scores = pd.DataFrame(sorted(weighted_scores.items(), key=lambda x:x[1], reverse=True))
        # print(weighted_scores)
        
        return X_train_transformed, X_test_transformed, weighted_scores

