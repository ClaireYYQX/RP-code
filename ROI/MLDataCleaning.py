import pandas as pd
import numpy as np

def remove_outliers(ROI_method):
    if ROI_method == 'JHU':
        DTI_df = JHU_get_metrics('DTI')
        FWE_df = JHU_get_metrics('FWE')
    
    DTI_df_transformed = data_transformation(DTI_df)
    FWE_df_transformed = data_transformation(FWE_df)

    DTI_indices = identify_outliers(DTI_df_transformed)
    FWE_indices = identify_outliers(FWE_df_transformed)
    common_indices = DTI_indices.intersection(FWE_indices)
    
    
    DTI_df = DTI_df.iloc[common_indices]
    FWE_df = FWE_df.iloc[common_indices]
    
    DTI_df.to_csv(f'data/ROI_{ROI_method}_DTI_df_4IQR.csv', index=True)
    FWE_df.to_csv(f'data/ROI_{ROI_method}_FWE_df_4IQR.csv', index=True)
    print(DTI_df.shape)
    print(FWE_df.shape)
    
def JHU_get_metrics(modality):
    if modality == 'DTI':
        file_name_list = ['mean_dti_SS_FA_JHU_thr07',
                          'mean_dti_SS_MD_JHU_thr07']
    elif modality == 'FWE':
        file_name_list = ['mean_fw_MS_FW_JHU_thr07',
                          'mean_fw_MS_MD_JHU_thr07',
                          'mean_fw_MS_FA_JHU_thr07']
            
    file_path = '/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/data/data_thr07/ROI_analysis/'

    with open('/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/data/data_thr07/ROI_analysis/JHU-labels.txt', 'r') as file:
        ROIs = [line.strip() for line in file]
    
    ROIs.pop()
    
    df_dict = dict()
    for file_name in file_name_list:
        df_dict[file_name] = pd.read_csv(file_path+file_name+'.csv', header=None)
        metric = file_name.split('_')[3]
        column_names = [metric+'_'+ROI_name for ROI_name in ROIs]
        df_dict[file_name].columns = column_names

    df_dict['participants'] = pd.read_excel('/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/participant_data.xlsx', sheet_name='fwDTI subset', engine='openpyxl')
    df_dict['participants'] = df_dict['participants'].drop(['Unnamed: 6', 'Unnamed: 7'], axis=1)
    df_metrics = pd.concat([df_dict[file_name] for file_name in file_name_list], axis=1)
    
    df = pd.concat([df_dict['participants']['age'], df_metrics], axis=1, sort=False)
    
    if modality == 'FWE':
        df = df.drop(['MD_Fornix_(column_and_body)', 'MD_Tapetum_L', 'MD_Tapetum_R', 'MD_Genu_of_corpus_callosum'], axis=1)
    print('a')
    
    df.to_csv(f'data/ROI_JHU_{modality}_df.csv', index=True)
    
    return df
        
# Function to calculate skewness estimate using quantiles
def get_quartile_skewness(column):
    Q1 = column.quantile(0.25)
    Q2 = column.quantile(0.5)  # Median
    Q3 = column.quantile(0.75)
    return (Q1 - 2*Q2 + Q3) / (Q3 - Q1)

def data_transformation(df):
    threshold = 0.1
            
    # Calculate quartile skewness
    quartile_skewness = df.apply(get_quartile_skewness)
    print(quartile_skewness.sort_values(ascending=True))
        
    df = df.apply(lambda x: np.log(x) if get_quartile_skewness(x) > threshold else x)
    df = df.apply(lambda x: -np.log(1-x) if get_quartile_skewness(x) < -threshold else x)
    quartile_skewness = df.apply(get_quartile_skewness)
    print(quartile_skewness.sort_values(ascending=True))
    return df
        
def identify_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # Condition for finding values within the acceptable range (not outliers)
    condition = ~((df < (Q1 - 4* IQR)) | (df > (Q3 + 4 * IQR))).any(axis=1)
    df_clean = df[condition]  # Apply condition
    return df[condition].index

remove_outliers('JHU')