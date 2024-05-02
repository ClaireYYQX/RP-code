import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
from collections import defaultdict
import pandas as pd
import sys
import nibabel as nib
import numpy as np

# Define file paths
selected_voxels_path = 'heatmap/heatmap_FWE_md_subsample_all.nii'
jhu_atlas_path = 'JHU-ICBM-labels-1mm.nii'
another_atlas_path = 'JHU-ICBM-tracts-maxprob-thr25-1mm.nii' 

# Load selected voxels image
selected_voxels_img = nib.load(selected_voxels_path)
selected_voxels_data = selected_voxels_img.get_fdata()

# Load the JHU atlas
jhu_atlas_img = nib.load(jhu_atlas_path)
jhu_atlas_data = jhu_atlas_img.get_fdata()

# Load the second atlas
another_atlas_img = nib.load(another_atlas_path)
another_atlas_data = another_atlas_img.get_fdata()

# Prepare the ROI names list from JHU atlas
label_file = '/imaging/correia/users/mc04/MRI_METHODS/FreeWaterDiffusion/data/data_thr07/ROI_analysis/JHU-labels.txt'  # Replace with your label file path
with open(label_file, 'r') as file:
    jhu_ROIs = [line.strip() for line in file]
    
other_ROIs =['Anterior_thalamic_radiation_L', 'Anterior thalamic radiation R',
'Corticospinal tract L', 'Corticospinal tract R', 'Cingulum (cingulate gyrus) L',
'Cingulum (cingulate gyrus) R', 'Cingulum (hippocampus)_L', 'Cingulum (hippocampus) R',
'Forceps major', 'Forceps minor', 'Inferior fronto-occipital fasciculus L',
'Inferior fronto-occipital fasciculus R', 'Inferior longitudinal fasciculus L',
'Inferior longitudinal fasciculus R', 'Superior longitudinal fasciculus L',
'Superior longitudinal fasciculus R', 'Uncinate fasciculus L',
'Uncinate fasciculus R', 'Superior longitudinal fasciculus (temporalpart) L',
'Superior longitudinal fasciculus (temporal part) R']

jhu_ROIs.pop()
ROIs = jhu_ROIs + other_ROIs + ['Non_ROI']

def calculate_intensities(atlas_data, selected_voxels_data, atlas_name):
    unique_roi_labels = np.unique(atlas_data[atlas_data > 0])
    roi_intensity_sums = {}
    for roi_label in unique_roi_labels:
        roi_mask = atlas_data == roi_label
        roi_voxels_intensities = selected_voxels_data[roi_mask]
        
        new_label = f"{atlas_name}_{roi_label}"
        roi_intensity_sums[new_label] = np.sum(roi_voxels_intensities)
    return roi_intensity_sums

# Calculate intensities for both atlases
jhu_intensities = calculate_intensities(jhu_atlas_data, selected_voxels_data, 'JHU')
another_intensities = calculate_intensities(another_atlas_data, selected_voxels_data, 'Another')

combined_intensities = {**jhu_intensities, **another_intensities}

non_roi_mask = np.logical_and(jhu_atlas_data == 0, another_atlas_data == 0)
non_roi_voxels_intensities = selected_voxels_data[non_roi_mask]
combined_intensities['Non_ROI'] = np.sum(non_roi_voxels_intensities)

roi_intensity_df = pd.DataFrame(list(combined_intensities.items()), columns=['ROI_Label', 'Intensity_Sum'])
print(roi_intensity_df)
roi_intensity_df['ROI'] = ROIs
print(roi_intensity_df)
roi_intensity_df = roi_intensity_df.sort_values(by='Intensity_Sum', ascending=False)

roi_intensity_df.to_csv('ROI_ranking_analysis/intensity_combined_FWE_md_subsample_all.csv', index=False)

print(roi_intensity_df)
