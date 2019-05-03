from __future__ import division
import sys
import numpy as np
from glob import glob
from os.path import join, basename, exists
from nilearn import input_data
from sklearn.preprocessing import normalize
from nilearn import plotting
from nilearn.image import concat_imgs, mean_img, resample_to_img

data_dir = '/scratch/kbott/hcp7t'
roi_dir = '/home/kbott006/hcp7t/habenulas'
sink_dir = '/home/kbott006/hcp7t/habenula-rsfc'

#subjects = ['130114', '128935', '130518']
subjects = sys.argv[1:]
print subjects

runs = ['rfMRI_REST1_7T_PA_hp2000_clean.nii.gz',
        'rfMRI_REST2_7T_AP_hp2000_clean.nii.gz',
        'rfMRI_REST3_7T_PA_hp2000_clean.nii.gz',
        'rfMRI_REST4_7T_AP_hp2000_clean.nii.gz']

for s in subjects:
    level = 'subject-level'
    zmaps = []
    rmaps = []
    print(s)
    for run in np.arange(0,4):
        try:
            #concatenate 4 runs into a mega run
            fmri_file = join(data_dir, s, runs[run])

            #confound = join(data_dir, s, 'session-0', 'resting-state', 'resting-state-0', 'nakatomi1.feat', 'mc', 'prefiltered_func_data_mcf.par')
            #hypothal_roi = join(roi_dir, s, 'roi', 'hypothal-func.nii.gz')
            hb_func = join(roi_dir,'{0}-hb-kb.nii.gz'.format(s))
            hb_roi_qc = join(sink_dir, 'roi-qc', '{0}-{1}-hb-func.png'.format(s,run))
            avg_func = join(sink_dir, 'mean_func.nii.gz')
            hb = join(roi_dir,'{0}-{1}-hb-resampled.nii.gz'.format(s,run))
            mean_func = mean_img(fmri_file).to_filename(avg_func)

            hb_resampled = resample_to_img(hb_func, avg_func, interpolation='nearest')
            hb_resampled.to_filename(hb)

            plotting.plot_roi(hb, avg_func, output_file=hb_roi_qc)

            roi_masker = input_data.NiftiMasker(hb,detrend=False, standardize=True, t_r=1.)
            roi_timeseries = roi_masker.fit_transform(fmri_file)
            print 'extracted hb timeseries'
            brain_masker = input_data.NiftiMasker(detrend=False, standardize=True, t_r=1.,
                                                  smoothing_fwhm=3.)
            brain_timeseries = brain_masker.fit_transform(fmri_file)
            print 'extracted brain timeseries'

            hb_ts = np.mean(roi_timeseries, axis=1)
            #hb = roi_timeseries[:,1]
            print hb_ts.shape

            sbc_hb = np.dot(brain_timeseries.T, hb_ts) / hb_ts.shape[0]
            print 'dot product is done'
            #maybe save the z-image and run randomise on that?
            sbc_hb_z = np.arctanh(sbc_hb)
            print 'we have z values'
            sbc_hb_img = brain_masker.inverse_transform(sbc_hb.T)
            sbc_hb_z_img = brain_masker.inverse_transform(sbc_hb_z.T)
            print 'and a z-value image'

            output_png = join(sink_dir, level, '{0}-{1}_sbc_hb.png'.format(s,run))
            output_nii = join(sink_dir, level, '{0}-{1}_sbc_hb.nii.gz'.format(s,run))
            output_z_nii = join(sink_dir, level, '{0}-{1}_sbc_z_hb.nii.gz'.format(s,run))

            rmaps.append(output_nii)
            zmaps.append(output_z_nii)
            sbc_hb_img.to_filename(output_nii)
            sbc_hb_z_img.to_filename(output_z_nii)
            plotting.plot_stat_map(sbc_hb_img, bg_img=mean_func, output_file=output_png)
        except Exception as e:
            print(e)
    try:
        avg_z_map = mean_img(zmaps)
        avg_z_map.to_filename(join(sink_dir, level, '{0}_mean_zmap.nii.gz'.format(s)))
    except Exception as e:
        print(e)
