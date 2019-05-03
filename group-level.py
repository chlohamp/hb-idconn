import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join, exists
from nipype.interfaces.fsl import Randomise
from nilearn.image import concat_imgs, smooth_img

import datetime
today = datetime.date.today()

def func_to_mni(subject, reg_dir, data_dir, qc_dir, sink_dir, standard, qc):
    from nipype.interfaces.fsl import ApplyXFM
    from nipype.interfaces.utility import Function
    from os.path import join

    def get_niftis(subject_id, data_dir, reg_dir, standard):
        from os.path import join, exists
        map = join(data_dir, '{0}_sbc_hb.nii.gz'.format(subject_id))
        xfm = join(reg_dir, 'example_func2standard.mat')
        assert exists(map), "map does not exist"
        assert exists(xfm), "xfm does not exist"
        if standard == 'mni':
            template = '/home/applications/fsl/5.0.8/data/standard/MNI152_T1_2mm_brain.nii.gz'
        else:
            template = standard
        return map, xfm, template

    data = Function(function=get_niftis, input_names=['subject_id', 'data_dir', 'reg_dir', 'standard'],
                            output_names=['map', 'xfm', 'template'])
    data.inputs.data_dir = data_dir
    data.inputs.subject_id = subject
    data.inputs.reg_dir = reg_dir
    data.inputs.standard = standard
    grabber = data.run()

    applyxfm = ApplyXFM()
    applyxfm.inputs.in_file = grabber.outputs.map
    applyxfm.inputs.in_matrix_file = grabber.outputs.xfm
    applyxfm.inputs.out_file = join(sink_dir, '{0}_mni.nii.gz'.format(subject))
    applyxfm.inputs.reference = grabber.outputs.template
    applyxfm.inputs.apply_xfm = True
    result = applyxfm.run()

    if qc == True:
        from nilearn.plotting import plot_stat_map
        plot_stat_map(result.outputs.out_file, grabber.outputs.template,
                      output_file=join(qc_dir, '{0}_mni.png'))
    else:
        return result.outputs.out_file

#roi_dir = '/home/kbott006/auburn7t/'
#data_dir = '/home/kbott006/auburn7t/results-11-19-18'
#qc_dir = '/home/kbott006/auburn7t/reg-qc'
#sink_dir = '/home/kbott006/auburn7t/results-11-19-18/lvl2'

data_dir = '/home/kbott006/hcp7t/habenula-rsfc/subject-level'
sink_dir = '/home/kbott006/hcp7t/habenula-rsfc/group-level'

#parent_dir = '/Users/Katie/Dropbox/Data/habenula/derivatives/hb_test/'
#subjects = ['HIP001', 'HIP002', 'HIP003', 'HIP004', 'HIP005',
#            'HIP006', 'HIP008', 'HIP009', 'HIP016',
#            'HIP011', 'HIP013', 'HIP014', 'HIP015',
#            'HIP018', 'HIP021', 'HIP022', 'HIP027',
#            'HIP023', 'HIP024', 'HIP025', 'HIP026',
#            'HIP030', 'HIP031', 'HIP033', 'HIP034']

subjects = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '118225', '125525', '126426', '126931', '128935', '130114', '130518', '131217', '131722', '132118', '134627', '134829', '135124', '137128', '140117', '144226', '145834', '146129', '146432', '146735', '146937', '148133', '155938', '156334', '157336', '158035', '158136', '159239', '162935', '164131', '164636', '167036', '169343', '995174', '971160', '966975', '958976', '951457', '943862', '942658', '927359', '926862', '910241', '905147', '901442', '901139', '899885', '898176', '878877', '878776', '872764', '871762', '861456', '814649', '789373', '783462', '782561', '771354', '770352', '765864', '757764', '751550', '601127', '617748', '627549', '638049', '644246', '654552', '671855', '680957', '690152', '706040', '724446', '725751', '732243', '745555', '581450', '573249', '572045', '562345', '550439', '547046', '541943', '525541', '467351']

for subject in subjects:
    assert exists(join(data_dir,'{0}_mean_zmap.nii.gz'.format(subject))), '{0}_mean_zmap.nii.gz does not exist'.format(subject)

runs = np.arange(0,4)
sm_fwhm = 0.
maps = []

for subject in subjects:
    if sm_fwhm > 0:
        sm_map = smooth_img(join(data_dir,'{0}_mean_zmap.nii.gz'.format(subject)), sm_fwhm)
        sm_map.to_filename(join(data_dir, '{0}_mean_zmap_sm-{1}mm.nii.gz'.format(subject, sm_fwhm)))
        maps.append(sm_map)
    else:
        maps.append(join(data_dir,'{0}_mean_zmap.nii.gz'.format(subject)))
    #for run in runs:
        #reg_dir = '/home/data/nbc/auburn/data/pre-processed/{0}/session-0/resting-state/resting-state-0/nakatomi1.feat/reg'.format(subject)
        #xfmd_map = func_to_mni(subject, reg_dir, data_dir, qc_dir, sink_dir, standard='mni', qc=False)
        #xfmd_map = join(data_dir, '{0}-{1}_sbc_hb.nii.gz'.format(subject, run))
        #smooth these xfmd maps so that the results don't look so much like spots
        #sm_xfmd_map = smooth_img(xfmd_map, sm_fwhm)
        #maps.append(xfmd_map)
all_maps = concat_imgs(maps)
all_maps_file = join(sink_dir, '4d-{0}_subj-{1}.nii'.format(len(subjects), today))
all_maps.to_filename(all_maps_file)

randomise = Randomise(one_sample_group_mean=True, tfce=True)
randomise.inputs.in_file = join(sink_dir, all_maps_file)
randomise.inputs.base_name = join(sink_dir,'hcp_{0}_subj-{1}_sm-{2}mm'.format(len(subjects), today, sm_fwhm))

results = randomise.run()
