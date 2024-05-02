import pandas as pd
import sys

random_seed = 2
df = pd.read_excel('/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/participant_data.xlsx', sheet_name='fwDTI subset', engine='openpyxl')
df_1 = pd.read_csv(f'ROI_JHU_DTI_df_4IQR.csv')
print(df_1)
index = df_1['Unnamed: 0']
df_cleaned = df.iloc[index].reset_index(drop=True)
print(df_cleaned)
df = pd.concat([df_1, df_cleaned.loc[:, ['hand','gender_code']]], axis=1, sort=False)
print(df)
df_filtered = df.loc[(df['hand']>0) & (df['age']>=20) & (df['age']<=85) & (df['gender_code']==1)]

target_column = 'age'
age_bins = [18, 20, 30, 40, 50, 60, 70, 80, 85, 90]
age_labels = ['18-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80','80-85', '85-90']
df_filtered['age_group'] = pd.cut(df_filtered['age'], bins=age_bins, labels=age_labels, right=False)

# samples_per_group = {
#     '20-30': 14,
#     '30-40': 4,
#     '40-50': 10,
#     '50-60': 9,
#     '60-70': 34, 
#     '70-80': 15,
#     '80-85': 3
# }

samples_per_group = {
    '20-30': 14,
    '30-40': 14,
    '40-50': 14,
    '50-60': 14,
    '60-70': 14, 
    '70-80': 13,
    '80-85': 6
}


def sample_by_group(x):
    group = x.name
    n_samples = samples_per_group.get(group, 0)
    return x.sample(n=n_samples, random_state=random_seed)


sampled_df = df_filtered.groupby('age_group').apply(sample_by_group)
df = sampled_df.drop(['hand', 'gender_code', 'age_group'], axis=1)
print(df)
df = df.reset_index()
df['Unnamed: 0'].to_csv(f'subsample/subsample_index_even_{random_seed}.csv', index=False)
# df.to_csv('DTI_subsample.csv', index=True)
# df = pd.read_csv
