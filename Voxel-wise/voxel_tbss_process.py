import pandas as pd
from sklearn.preprocessing import StandardScaler

class MLDataProcessor:
    def __init__(self, modality):
        self.df = None
        self.modality = modality

    def z_score_normalisation(self, X_train, X_test):
        columns = X_train.columns
        
        # Creating a StandardScaler object
        scaler = StandardScaler()

        # Fitting and transforming the training data
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columns)

        #  Only transforming the test data
        X_test = pd.DataFrame(scaler.transform(X_test), columns=columns)
        
        columns=X_train.columns

        return X_train, X_test
    
    def get_age_groups(self, X_train, Y_train):
        df = pd.concat([X_train, Y_train], axis=1)
        age_bins = [18, 20, 30, 40, 50, 60, 70, 80, 90]
        age_labels = ['18-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80','80-90']
        
        # age_bins = [20, 40, 60, 70, 85]
        # age_labels = ['20-40', '40-60', '60-70', '70-85']
        
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        Y_group = df['age_group']
        
        return Y_group