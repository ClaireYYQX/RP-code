import numpy as np
import nibabel as nib
import sys
from nilearn import plotting, image
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Load reference brain image
file_path = '/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/data/data_thr07/FA/'
ref_img = nib.load(f'{file_path}dtifit_wls_FA_sub-CC420217_FA_to_target.nii.gz')

# Get the dimensions of the reference image
dims = ref_img.shape    
print(dims)

# Initialise a 3D volume for the heatmap
heatmap_volume = np.zeros(dims)

def get_df(model, i, j):
    df = pd.read_csv(f'voxel_selected_voxels/{model}_{i}_fold{j}_tbss-new.csv')
    df_md = df[df['cluster_id'].str.startswith('MD')]
    df_fa = df[df['cluster_id'].str.startswith('FA')]
    if model == 'FWE':
        df_fw = df[df['cluster_id'].str.startswith('FW')]
        return df_md, df_fa, df_fw
    else:
        return df_md, df_fa

def get_count(df):
    return df.shape[0]
    
def get_voxel_coords(df):
    df_coords = df[['x','y','z']]
    voxel_coords = list(df_coords.itertuples(index=False, name=None))
    return voxel_coords

voxel_count = defaultdict(int)
voxel_coords = defaultdict(list)
for i in range(41, 51):
    for j in range(1, 11):
        df_md_DTI, df_fa_DTI = get_df('DTI', i, j)
        df_md_FWE, df_fa_FWE, df_fw_FWE = get_df('FWE', i, j)
        voxel_coords['DTI-md'] += get_voxel_coords(df_md_DTI)
        voxel_coords['DTI-fa'] += get_voxel_coords(df_fa_DTI)
        voxel_coords['FWE-md'] += get_voxel_coords(df_md_FWE)
        voxel_coords['FWE-fa'] += get_voxel_coords(df_fa_FWE)
        voxel_coords['FWE-fw'] += get_voxel_coords(df_fw_FWE)
        
        voxel_count['DTI-md'] += get_count(df_md_DTI)
        voxel_count['DTI-fa'] += get_count(df_fa_DTI)
        voxel_count['FWE-md'] += get_count(df_md_FWE)
        voxel_count['FWE-fa'] += get_count(df_fa_FWE)
        voxel_count['FWE-fw'] += get_count(df_fw_FWE)


def get_heatmap(coords, model, modality):
    # Initialise a 3D volume for the heatmap
    heatmap_volume = np.zeros(dims)
    
    # Increment the value at each voxel coordinate
    for coord in coords:
        heatmap_volume[coord] += 1

    # Normalize the heatmap
    heatmap_volume = (heatmap_volume - heatmap_volume.min()) / (heatmap_volume.max() - heatmap_volume.min())

    # Create a NIfTI file for the heatmap
    heatmap_img = nib.Nifti1Image(heatmap_volume, affine=ref_img.affine)
    nib.save(heatmap_img, f'heatmap/heatmap_{model}_{modality}_tbss-new.nii')

    # Load statistical map
    stat_map_img = f'heatmap/heatmap_{model}_{modality}_tbss-new.nii'

    # Load the brain template
    brain_template = f'{file_path}dtifit_wls_FA_sub-CC420217_FA_to_target.nii.gz'

    # Plotting the heatmap overlay
    display = plotting.plot_glass_brain(
        stat_map_img,
        display_mode='ortho', 
        colorbar=True, 
        threshold=0,  
        cmap= plt.cm.hot_r,  
        black_bg=False,   
        plot_abs=False
    )

    display.savefig(f'heatmap/heatmap_{model}_{modality}_tbss-new.png')

    display.close()
    
print(voxel_count)
 
get_heatmap(voxel_coords['DTI-fa'],'DTI','fa')
get_heatmap(voxel_coords['DTI-md'],'DTI','md')
get_heatmap(voxel_coords['FWE-fa'],'FWE','fa')
get_heatmap(voxel_coords['FWE-md'],'FWE','md')
get_heatmap(voxel_coords['FWE-fw'],'FWE','fw')