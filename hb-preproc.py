import os.path import join, basename, exists
import numpy as np
import nibabel as nb
from glob import glob
from bids.grabbids import BIDSLayout
from nipype import MapNode, JoinNode
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces import fsl
from IPython.display import Image
from bids.grabbids import BIDSLayout
from os.path import join, basename
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
import nipype.algorithms as na
from IPython.display import Image

def get_niftis(subject_id, data_dir):
    from glob import glob
    from os.path import join, basename, exists
    parent_dir = '/Users/Katie/Dropbox/Data/habenula/'
    t1 = join(parent_dir, subject_id, 'anat', '3d.nii')
    epi = join(parent_dir, subject_id, 'func', 'rest.nii')
    assert exists(t1), "t1 does not exist"
    assert exists(epi), "epi does not exist"
    return t1, epi

def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative

    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params2, fmt=str('%.5f'))
        out_files.append(filename)
    return out_files

def build_filter(motion_params, comp_norm, outliers, detrend_poly=None):
    """Builds a regressor set comprising motion parameters, composite norm and
    outliers

    The outliers are added as a single time point column for each outlier


    Parameters
    ----------

    motion_params: a text file containing motion parameters and its derivatives
    comp_norm: a text file containing the composite norm
    outliers: a text file containing 0-based outlier indices
    detrend_poly: number of polynomials to add to detrend

    Returns
    -------
    components_file: a text file containing all the regressors
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
        out_params = np.hstack((params, norm_val[:, None]))
        if detrend_poly:
            timepoints = out_params.shape[0]
            X = np.ones((timepoints, 1))
            for i in range(detrend_poly):
                X = np.hstack((X, legendre(
                    i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
            out_params = np.hstack((out_params, X))
        try:
            outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((out_params.shape[0], 1))
            outlier_vector[index] = 1
            out_params = np.hstack((out_params, outlier_vector))
        filename = os.path.join(os.getcwd(), "filter_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params, fmt=str('%.5f'))
        out_files.append(filename)
    return out_files

def bandpass_filter(files, lowpass_freq, highpass_freq, fs):
    """Bandpass filter the input files

    Parameters
    ----------
    files: list of 4d nifti files
    lowpass_freq: cutoff frequency for the low pass filter (in Hz)
    highpass_freq: cutoff frequency for the high pass filter (in Hz)
    fs: sampling rate (in Hz)
    """
    out_files = []
    for filename in filename_to_list(files):
        path, name, ext = split_filename(filename)
        out_file = os.path.join(os.getcwd(), name + '_bp' + ext)
        img = nb.load(filename)
        timepoints = img.shape[-1]
        F = np.zeros((timepoints))
        lowidx = timepoints/2 + 1
        if lowpass_freq > 0:
            lowidx = np.round(lowpass_freq / fs * timepoints)
        highidx = 0
        if highpass_freq > 0:
            highidx = np.round(highpass_freq / fs * timepoints)
        F[highidx:lowidx] = 1
        F = ((F + F[::-1]) > 0).astype(int)
        data = img.get_data()
        if np.all(F == 1):
            filtered_data = data
        else:
            filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
        img_out = nb.Nifti1Image(filtered_data, img.get_affine(),
                                 img.get_header())
        img_out.to_filename(out_file)
        out_files.append(out_file)
    return list_to_filename(out_files)

hb_test_wf = pe.Workflow(name="hb_test", base_dir="/Users/Katie/Dropbox/Data/habenula/derivatives")

subjects = ['HIP001', 'HIP001','HIP003']
subj_iterable = pe.Node(IdentityInterface(fields=['subject_id']),
                        name='subj_iterable')
subj_iterable.iterables = ('subject_id', subjects)

DataGrabber = pe.Node(Function(function=get_niftis,
                               input_names=["subject_id", "data_dir"],
                               output_names=["T1", "bold"]),
                         name="BIDSDataGrabber")
DataGrabber.inputs.data_dir = "/Users/Katie/Dropbox/Data/habenula/"
#DataGrabber.inputs.subject_id = subject


moco = pe.Node(fsl.MCFLIRT(cost='normmi'),
                  name="mcflirt")
extractb0 = pe.Node(fsl.ExtractROI(t_size=1, t_min=1),
                       name = "extractb0")
bet = pe.Node(fsl.BET(frac=0.1, mask=True),
                 name="bet_func")
bet2 = pe.Node(fsl.BET(frac=0.1),
                 name="bet_struc")
segment = pe.Node(fsl.FAST(out_basename='fast_'),
                     name="fastSeg")
flirting = pe.Node(fsl.FLIRT(cost_func='normmi', dof=7, searchr_x=[-180, 180],
                             searchr_y=[-180, 180], searchr_z=[-180,180]),
                   name="struc_2_func")
applyxfm = pe.MapNode(fsl.ApplyXfm(apply_xfm = True),
                      name="MaskEPI", iterfield=['in_file'])
erosion = pe.MapNode(fsl.ErodeImage(),
                     name="erode_masks", iterfield=['in_file'])
regcheckoverlay = pe.Node(fsl.Overlay(auto_thresh_bg=True, stat_thresh=(100,500)),
                         name='OverlayCoreg')
regcheck = pe.Node(fsl.Slicer(),
                  name='CheckCoreg')
#filterfeeder = pe.MapNode(fsl.ImageMeants(eig=True, ))

datasink = pe.Node(nio.DataSink(),
                   name='datasink')
datasink.inputs.base_directory = "/Users/Katie/Dropbox/Data/habenula/derivatives/hb_test"

# Connect alllllll the nodes!!
hb_test_wf.connect(subj_iterable, 'subject_id', DataGrabber, 'subject_id')
hb_test_wf.connect(DataGrabber, 'bold', moco, 'in_file')
hb_test_wf.connect(moco, 'out_file', extractb0, 'in_file')
hb_test_wf.connect(extractb0, 'roi_file', bet, 'in_file')
hb_test_wf.connect(bet, 'out_file', datasink, '@epi_brain')
hb_test_wf.connect(DataGrabber, 'T1', bet2, 'in_file')
hb_test_wf.connect(bet2, 'out_file', flirting, 'in_file')
hb_test_wf.connect(bet, 'out_file', flirting, 'reference')
hb_test_wf.connect(bet, 'mask_file', datasink, '@func_brain_mask')
hb_test_wf.connect(flirting, 'out_matrix_file', applyxfm, 'in_matrix_file')
hb_test_wf.connect(bet2, 'out_file', segment, 'in_files')
hb_test_wf.connect(segment, 'partial_volume_files', applyxfm, 'in_file')
hb_test_wf.connect(bet, 'out_file', applyxfm, 'reference')
hb_test_wf.connect(moco, 'out_file', datasink, '@motion_corrected')
hb_test_wf.connect(applyxfm, 'out_file', erosion, 'in_file')
hb_test_wf.connect(erosion, 'out_file', datasink, '@eroded_mask')
hb_test_wf.connect(flirting, 'out_file', regcheckoverlay, 'background_image')
hb_test_wf.connect(bet, 'out_file', regcheckoverlay, 'stat_image')
hb_test_wf.connect(regcheckoverlay, 'out_file', regcheck, 'in_file')
hb_test_wf.connect(regcheck, 'out_file', datasink, '@regcheck')

hb_test_wf.run()
hb_test_wf.write_graph()
