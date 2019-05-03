
# coding: utf-8

# In[14]:


import neurosynth as ns
from neurosynth.base.dataset import Dataset
from neurosynth.analysis import decode, meta
from datetime import datetime
from os.path import join, basename
from nilearn.image import resample_to_img
from nilearn.plotting import plot_glass_brain
from glob import glob
import nibabel as nib
import numpy as np
import pickle
import pandas as pd


# In[7]:


map_dir = '/Users/Katie/Dropbox/Projects/habenula/resting-state/decoding'
sink_dir = '/Users/Katie/Dropbox/Projects/habenula/resting-state/decoded'

mni152_2mm = '/Users/Katie/Dropbox/Data/templates/MNI152_T1_2mm_brain.nii.gz'

roi_files = ['osl_tscores_hbfc_DDisc_SV_6mo_200-thresh_gt_2.nii.gz','osl_tscores_hbfc_ER40HAP-alpha_thresh_gt_2.nii.gz', 'hcp_96_subj-2019-04-30_sm-0.0mm_tstat1-thresh_gt_9.5.nii.gz']

mask_names = []
for i in np.arange(0,len(roi_files)):
    roi = join(map_dir, roi_files[i])
    file = roi.strip('.nii.gz')
    mask_names.append(basename(file))
    mask_img = nib.load(roi)
    mask_img_data = mask_img.get_fdata()
    if mask_img_data.shape != (91, 109, 91):
        resampled_roi = resample_to_img(roi, mni152_2mm, interpolation='nearest', copy=True)
        resampled_file = join(map_dir, '{0}_mni2mm.nii.gz'.format(file))
        resampled_roi.to_filename(resampled_file)
        roi_files[i] = resampled_file
        plot_glass_brain(resampled_file, output_file=join(map_dir, '{0}_mni2mm.png'.format(basename(file))))

# In[16]:

print('loading dataset...')
tds = datetime.now()
dataset = Dataset('/Users/Katie/Dropbox/Data/neurosynth-v0.7/database.txt')
dataset.add_features('/Users/Katie/Dropbox/Data/neurosynth-v0.7/features.txt')
tdf = datetime.now()

print('dataset loaded! only took {0}'.format((tdf-tds)))

for i in np.arange(0,len(mask_names)):
    print('{0}\nmeta-analyzing {1}...'.format(datetime.now(), mask_names[i]))
    tmas = datetime.now()
    ids = dataset.get_studies(mask=roi_files[i], )
    ma = meta.MetaAnalysis(dataset, ids)
    ma.save_results(output_dir=sink_dir, prefix=mask_names[i], image_list=['association-test_z', 'association-test_z_FDR_0.01'])
    tmaf = datetime.now()
    print('meta-analysis took {0}\ndecoding {1}...'.format((tmaf-tmas),mask_names[i]))
    decoder = decode.Decoder(dataset, image_type='association-test_z')
    result = decoder.decode([join(sink_dir, '{0}_association-test_z.nii.gz'.format(mask_names[i]))],
                            save=join(sink_dir,'decoded_{0}.txt'.format(mask_names[i])))
    tdcf = datetime.now()
    print('decoding {0} took {1}'.format(mask_names[i], (tdcf-tmaf)))
