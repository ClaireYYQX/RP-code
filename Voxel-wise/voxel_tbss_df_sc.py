from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
from collections import defaultdict

def cluster_voxels(modality, coordinates_df, minimum_extent=100, distances=[20, 25, 30], n_passes=3):
    if len(distances) != n_passes:
        raise Warning("You have not specified the correct number of distances")

    for index, distance in enumerate(distances):
        print(f"Pass {index + 1}")

        # Check for valid ranges in the coordinates
        if any(coordinates_df[col].max() - coordinates_df[col].min() == 0 for col in coordinates_df.columns):
            break

        # Clustering using KDTree
        tree = KDTree(coordinates_df)
        indices = tree.query_radius(coordinates_df, r=distance)

        # Assigning cluster IDs
        cluster_id = -np.ones(len(coordinates_df), dtype=int)
        cluster_counter = 0
        for i, ind in enumerate(indices):
            if cluster_id[i] == -1:
                cluster_id[ind] = cluster_counter
                cluster_counter += 1

        coordinates_df.loc[:, 'cluster_int_id'] = cluster_id

        # Calculating cluster extent and filtering
        cluster_extent = coordinates_df['cluster_int_id'].value_counts()
        large_clusters = cluster_extent[cluster_extent >= minimum_extent].index
        filtered_coordinates_df = coordinates_df[coordinates_df['cluster_int_id'].isin(large_clusters)]

        # Terminate if no small clusters
        if all(cluster_extent >= minimum_extent):
            break

    filtered_coordinates_df.loc[:, 'cluster_id'] = [modality + '_Cluster_' + str(id) for id in filtered_coordinates_df['cluster_int_id']]
    
    return filtered_coordinates_df

def feature_clustering(df, modality):
    coordinates = set()
    for column in df.columns:
        coord = tuple(map(int, column.split('_')))
        coordinates.add(coord)

    coordinates_df = pd.DataFrame(list(coordinates), columns=['x', 'y', 'z'])

    clustered_df = cluster_voxels(modality, coordinates_df)
    return clustered_df

def cluster_mean(modality, clustered_voxels_df, subjects_df):
    voxel_to_cluster = {f"{x}_{y}_{z}": cluster_id for x, y, z, cluster_id in clustered_voxels_df[['x', 'y', 'z', 'cluster_id']].to_numpy()}

    mean_per_cluster = pd.DataFrame(index=subjects_df.index)

    # cluster_dict = {}

    for cluster_id in clustered_voxels_df['cluster_id'].unique():
        voxel_cols = [col for col in subjects_df.columns if voxel_to_cluster.get(col) == cluster_id]
        # print(len(voxel_cols))
        mean_per_cluster[f'{cluster_id}'] = subjects_df[voxel_cols].mean(axis=1)
        # cluster_dict[f"{modality}_Cluster_{cluster_id}"] = voxel_cols
        
    return mean_per_cluster

def clustering_for_modality(directory, modality='FA', random_seed=100):
    data_dict = defaultdict(dict)
    cluster_dict = defaultdict(pd.DataFrame)
    k = 10000
    for fold in range(1,11):
        X_train = pd.read_csv(f"{directory}{modality}_{random_seed}_Xtrain_fold_{fold}_k{k}_tbss-new.csv")
        X_test = pd.read_csv(f"{directory}{modality}_{random_seed}_Xtest_fold_{fold}_k{k}_tbss-new.csv")
        Y_train = pd.read_csv(f"{directory}{modality}_{random_seed}_Ytrain_fold_{fold}_k{k}_tbss-new.csv")
        Y_test = pd.read_csv(f"{directory}{modality}_{random_seed}_Ytest_fold_{fold}_k{k}_tbss-new.csv")
        clustered_voxels_df = feature_clustering(X_train, modality)
        X_train_clustered = cluster_mean(modality, clustered_voxels_df, X_train)
        X_test_clustered = cluster_mean(modality, clustered_voxels_df, X_test)

        data_dict[str(fold)] = {'X_train': X_train_clustered, 'X_test' : X_test_clustered, 'Y_train':Y_train, 'Y_test':Y_test}
        cluster_dict[str(fold)] = clustered_voxels_df
        # print(X_train_clustered)
        # print(X_test_clustered)
        # print(Y_train)
        # print(Y_test)
        
    # print(cluster_dict)
    return data_dict, cluster_dict

def cfs(X_train, X_test, Y_train):
    selected_features = get_cfs_selected_features(X_train, Y_train)
    X_train = X_train.loc[:, selected_features]
    X_test = X_test.loc[:, selected_features]
    return X_train, X_test, selected_features

def get_cfs_selected_features(X_train, Y_train, threshold_features=0.7, threshold_target=0.3):
    corr_with_target = {}
    for col in X_train.columns:
        corr_with_target[col] = X_train[col].corr(Y_train.iloc[:, 0])

    corr_with_target = pd.Series(corr_with_target)
    # print(corr_with_target)

    feature_corr = X_train.corr()
    # print(feature_corr)

    selected_features = []
    # print(corr_with_target.abs().sort_values(ascending=False).index)

    for feature in corr_with_target.abs().sort_values(ascending=False).index:
        if np.abs(corr_with_target[feature]) >= threshold_target:
            if all(np.abs(feature_corr[feature][selected_feature]) < threshold_features for selected_feature in selected_features):
                selected_features.append(feature)

    # print(len(selected_features))
    return selected_features      

def merge_modalities(dict_list):
    merged_data_dict = defaultdict(dict)
    for fold in range(1, 11):
        X_train_list = []
        X_test_list = []
        for data_dict in dict_list:
            X_train_list.append(data_dict[str(fold)]['X_train'])
            X_test_list.append(data_dict[str(fold)]['X_test'])
        
        X_train = pd.concat(X_train_list, axis=1)
        X_test = pd.concat(X_test_list, axis=1)
        Y_train = data_dict[str(fold)]['Y_train']
        Y_test = data_dict[str(fold)]['Y_test']
        
        merged_data_dict[str(fold)] = {'X_train': X_train, 'X_test' : X_test, 'Y_train':Y_train, 'Y_test':Y_test} 
    
    return merged_data_dict

def apply_cfs(merged_data, MRI_model, random_seed, cluster_dict_list):
    for fold in range(1,11):
        cluster_df_list = [cluster_dict[str(fold)] for cluster_dict in cluster_dict_list]
        data = merged_data[str(fold)]
        # print(data)
        X_train, X_test, selected_features = cfs(data['X_train'], data['X_test'], data['Y_train'])
        merged_data[str(fold)]['X_train'], merged_data[str(fold)]['X_test'] = X_train, X_test
        merged_data[str(fold)]['selected_features'] = selected_features
        # print(X_train)
        # print(X_test)
        # print(selected_features)
        Y_train, Y_test = data['Y_train'], data['Y_test']
        
        directory = '/imaging/correia/wl05/final_scripts/Voxel/train_test_processed/'
        X_train.to_csv(f"{directory}{MRI_model}_{random_seed}_X_train_fold{str(fold)}_tbss-new.csv", index=False)
        X_test.to_csv(f"{directory}{MRI_model}_{random_seed}_X_test_fold{str(fold)}_tbss-new.csv", index=False)
        Y_train.to_csv(f"{directory}{MRI_model}_{random_seed}_Y_train_fold{str(fold)}_tbss-new.csv", index=False)
        Y_test.to_csv(f"{directory}{MRI_model}_{random_seed}_Y_test_fold{str(fold)}_tbss-new.csv", index=False)
            
        selected_coordinates = get_selected_voxels(selected_features, cluster_df_list)
        selected_coordinates.to_csv(f'/imaging/correia/wl05/final_scripts/Voxel/voxel_selected_voxels/{MRI_model}_{random_seed}_fold{str(fold)}_tbss-new.csv')
    
    return merged_data

def get_selected_voxels(selected_clusters, cluster_df_list):
    selected_voxels_list = []
    for cluster_df in cluster_df_list:
        # print(selected_clusters)
        # print(cluster_df.columns)

        cluster_df_selected = cluster_df[cluster_df['cluster_id'].isin(selected_clusters)] 
        # print(cluster_df_selected)  
        
        selected_voxels_list.append(cluster_df_selected)
        
    selected_voxels_df = pd.concat(selected_voxels_list)
        
    # print(selected_voxels_df)
    return selected_voxels_df
        
for i in range(41, 51):
    random_seed = i
    directory_DTI = '/imaging/correia/wl05/final_scripts/Voxel/train_test_data/DTI/'
    DTI_FA_dict, cluster_FA_dict= clustering_for_modality(directory_DTI, 'FA', random_seed)
    DTI_MD_dict, cluster_MD_dict = clustering_for_modality(directory_DTI, 'MD', random_seed)
    DTI_dict_list = [DTI_FA_dict, DTI_MD_dict]
    DTI_cluster_dict_list = [cluster_FA_dict, cluster_MD_dict]

    directory_FWE = '/imaging/correia/wl05/final_scripts/Voxel/train_test_data/FWE/'
    FWE_FA_dict, cluster_FA_fw_dict = clustering_for_modality(directory_FWE, 'FA', random_seed)
    FWE_FW_dict, cluster_FW_fw_dict= clustering_for_modality(directory_FWE, 'FW', random_seed)
    FWE_MD_dict, cluster_MD_fw_dict = clustering_for_modality(directory_FWE, 'MD', random_seed)
    FWE_dict_list = [FWE_FA_dict, FWE_FW_dict, FWE_MD_dict]
    FWE_cluster_dict_list = [cluster_FA_fw_dict, cluster_MD_fw_dict, cluster_FW_fw_dict]


    DTI_merged_data = merge_modalities(DTI_dict_list)
    FWE_merged_data = merge_modalities(FWE_dict_list)
    apply_cfs(DTI_merged_data, 'DTI', random_seed, DTI_cluster_dict_list)
    apply_cfs(FWE_merged_data, 'FWE', random_seed, FWE_cluster_dict_list)
