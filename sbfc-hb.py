from __future__ import division
import numpy as np
from glob import glob
from os.path import join, basename, exists
from nilearn import input_data
from sklearn.preprocessing import normalize
import nipype
from nilearn import plotting
from nilearn.image import concat_imgs, mean_img

parent_dir = '/home/data/nbc/auburn/data/pre-processed'
roi_dir = '/home/kbott006/auburn7t/'
sink_dir = '/home/kbott006/auburn7t/results-11-19-18'

#parent_dir = '/Users/Katie/Dropbox/Data/habenula/derivatives/hb_test/'
subjects = ['HIP001', 'HIP002', 'HIP003', 'HIP004', 'HIP005',
            'HIP006', 'HIP008', 'HIP009', 'HIP016',
            'HIP011', 'HIP013', 'HIP014', 'HIP015',
            'HIP018', 'HIP021', 'HIP022', 'HIP027',
            'HIP023', 'HIP024', 'HIP025', 'HIP026',
            'HIP030', 'HIP031', 'HIP033', 'HIP034']

#subjects = ['HIP003']

for s in subjects:
    print s
    try:
        fmri_file = join(parent_dir, s, 'session-0', 'resting-state', 'resting-state-0', 'nakatomi1.feat', 'filtered_func_data.nii.gz')
        confound = join(parent_dir, s, 'session-0', 'resting-state', 'resting-state-0', 'nakatomi1.feat', 'mc', 'prefiltered_func_data_mcf.par')
        #hypothal_roi = join(roi_dir, s, 'roi', 'hypothal-func.nii.gz')
        hb_func = join(parent_dir, s, 'session-0', 'anatomical', 'anatomical-0', 'hb-func.nii.gz')
        hb_roi_qc = join(roi_dir, 'roi-qc', '{0}-hb-func.png'.format(s))
        mean_func = mean_img(fmri_file)
        plotting.plot_roi(hb_func, mean_func, output_file=hb_roi_qc)

        roi_masker = input_data.NiftiMasker(hb_func,
                                           detrend=False, standardize=True, t_r=3.,
                                           memory='nilearn_cache', memory_level=1, verbose=0)
        roi_timeseries = roi_masker.fit_transform(fmri_file, confounds=confound)
        print 'extracted hb timeseries'
        brain_masker = input_data.NiftiMasker(
            detrend=False, standardize=True, t_r=3., smoothing_fwhm=3.,
            memory='nilearn_cache', memory_level=1, verbose=1)
        brain_timeseries = brain_masker.fit_transform(fmri_file,
                                                   confounds=confound)
        print 'extracted brain timeseries'

        hb = np.mean(roi_timeseries, axis=1)
        #hb = roi_timeseries[:,1]
        print hb.shape

        sbc_hb = np.dot(brain_timeseries.T, hb) / hb.shape[0]
        print 'dot product is done'
        #maybe save the z-image and run randomise on that?
        sbc_hb_z = np.arctanh(sbc_hb)
        print 'we have z values'
        sbc_hb_img = brain_masker.inverse_transform(sbc_hb.T)
        sbc_hb_z_img = brain_masker.inverse_transform(sbc_hb_z.T)
        print 'and a z-value image'

        output_png = join(sink_dir, '{0}_sbc_hb.png'.format(s))
        output_nii = join(sink_dir, '{0}_sbc_hb.nii.gz'.format(s))
        output_z_nii = join(sink_dir, '{0}_sbc_z_hb.nii.gz'.format(s))
        sbc_hb_img.to_filename(output_nii)
        sbc_hb_z_img.to_filename(output_z_nii)
        plotting.plot_stat_map(sbc_hb_img, bg_img=mean_func, output_file=output_png)
    except Exception as e:
        print e
