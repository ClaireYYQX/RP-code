import nibabel as nib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from skrebate import ReliefF

def get_participants_age():
    folder_path = '/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion'
    df = pd.read_excel(os.path.join(folder_path, 'participant_data.xlsx'), sheet_name='fwDTI subset', engine='openpyxl')
    age = df['age']
    return age

def get_feature_matrix(): 
    metric_list = ['FA']
    folder_path = '/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/data/data_thr07/stats'
    # img_fa = nib.load(os.path.join(folder_path, 'all_FA_skeletonised.nii.gz')).get_fdata()
    img_fa = nib.load(os.path.join(folder_path, 'all_FA.nii.gz')).get_fdata()
    flattened_fa = np.transpose(img_fa.flatten().reshape(-1, 620))
    print(flattened_fa.shape)
    feature_names = [str(i)+'_'+str(j)+'_'+str(k) for i in range(182) for j in range(218) for k in range(182)]
    df_fa = pd.DataFrame(flattened_fa, columns = feature_names)
    df_fa = df_fa.loc[:, df_fa.any()]
    print(df_fa.shape)
    df_fa = remove_outliers(df_fa)
    print(df_fa.shape)
    return df_fa

def remove_outliers(df):
    df_clean = identify_outliers(df)
    return df_clean

def get_quartile_skewness(column):
    Q1 = column.quantile(0.25)
    Q2 = column.quantile(0.5)
    Q3 = column.quantile(0.75)
    return (Q1 - 2*Q2 + Q3) / (Q3 - Q1)
        
def identify_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    condition = ~((df < (Q1 - 3 * IQR)) | (df > (Q3 + 3 * IQR))).any(axis=0)
    df_clean = df.loc[:, condition]
    return df_clean

def get_subsample_index(rs=30):
    subsample_index = pd.read_csv(f'subsample/subsample_index_even_{rs}.csv')['Unnamed: 0'].tolist()
    print(subsample_index)
    return subsample_index

def cv_stratify_split(modality, modality_matrix, Age, random_seed):
    target_column = 'age'
    
    # # Define age groups
    # age_bins = [18, 20, 30, 40, 50, 60, 70, 90]
    # age_labels = ['18-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-90']
    # df = pd.concat([modality_matrix, Age], axis=1)
    
    age_bins = [20, 40, 60, 70, 85]
    age_labels = ['20-40', '40-60', '60-70', '70-85']
    subsample_index = get_subsample_index()
    df = pd.concat([modality_matrix, Age], axis=1).iloc[subsample_index]
    
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    x = df.drop([target_column, 'age_group'], axis=1)
    y = df['age']
        
    folds_data = []
    stratify_values = df['age_group'].values.ravel()
    splitter = StratifiedKFold(n_splits=10, random_state=random_seed, shuffle=True)
    fold_count = 1
    
    for train_index, test_index in splitter.split(x, stratify_values):
        X_train, X_test = pd.DataFrame(x.iloc[train_index]), pd.DataFrame(x.iloc[test_index])
        Y_train, Y_test = pd.DataFrame(y.iloc[train_index]), pd.DataFrame(y.iloc[test_index])
        fold_data = {'x_train': X_train, 'x_test': X_test, 'y_train': Y_train, 'y_test': Y_test}
        folds_data.append(fold_data)
        X_train, X_test = variance_thresholding(X_train, X_test)
        k = 10000
        print(k)
        X_train, X_test, top_features = relief_feature_selection(X_train, X_test, Y_train, Y_test, k)
        save_folds_to_file(modality, X_train, 'Xtrain', fold_count, random_seed, k)
        save_folds_to_file(modality, X_test, 'Xtest', fold_count, random_seed, k)
        save_folds_to_file(modality, Y_train, 'Ytrain', fold_count, random_seed, k)
        save_folds_to_file(modality, Y_test, 'Ytest', fold_count, random_seed, k)
        fold_count+=1

    return folds_data


def variance_thresholding(X_train, X_test):
    variances = X_train.var()

    sorted_variances = np.sort(variances)

    kth_percentile_index = int(len(sorted_variances) * 0.25)

    threshold = sorted_variances[kth_percentile_index]

    var_threshold_selector = VarianceThreshold(threshold=threshold)
    var_threshold_selector.fit(X_train)
    X_train_reduced = var_threshold_selector.transform(X_train)
    X_test_reduced = var_threshold_selector.transform(X_test)
    X_train = pd.DataFrame(X_train_reduced, columns=X_train.columns[var_threshold_selector.get_support()])
    X_test = pd.DataFrame(X_test_reduced, columns=X_test.columns[var_threshold_selector.get_support()])
    return X_train, X_test
    
def relief_feature_selection(X_train, X_test, Y_train, Y_test, k=10000):
    print('a')
    relief = ReliefF(n_features_to_select=k, n_neighbors=10, n_jobs=-1)  # Set the number of neighbors
    print('b')
    relief.fit(X_train.values, Y_train.values.ravel())
    print('a')
    # weights = relief.feature_importances_
    print('c')
    
    # Get the indices of the top k features
    indices = np.argsort(relief.feature_importances_)[-k:]

    # Select the top k features from X_train and X_test
    X_train = X_train.iloc[:, indices]
    X_test = X_test.iloc[:, indices]
    top_features = relief.top_features_
    print('f')
    return X_train, X_test, top_features

def save_folds_to_file(modality, df, train_test, fold, random_seed, k):
    directory = '/imaging/correia/wl05/final_scripts/Voxel/train_test_data/DTI/'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    filename = f"{directory}{modality}_{random_seed}_{train_test}_fold_{fold}_k{str(k)}_subsample_all-30even-IQR.csv"
    df.to_csv(filename, index=False)

df_fa = get_feature_matrix()
Age = get_participants_age()
modality = 'FA'
random_seed = 41
for random_seed in range(41,51):
    print(random_seed)
    cv_stratify_split(modality, df_fa, Age, random_seed)
