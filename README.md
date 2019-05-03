# hb-idconn
Here are all the code necessary to reproduce the results presented in SANS 2019 poster entitled _Intrinsic connectivity of the human habenula and its relation to
negative affect_. Data included ultra-high resolution, 7T resting-state fMRI (rsfMRI) data from (1) a 25-subject group, collected at Auburn University, and (2) 96 subjects from the Human Connectome Project with complete 7T rsfMRI data.\
Habenula ROIs were manually delineated on a per-subject basis, on subjects' T1-weighted images, based on the procedures outlined by Lawson et al. (2013).
## 1. Replication of Torrisi et al. (2017) in a similar, yet independent 7T rsfMRI dataset from Auburn University
`hb_preproc.py` performs necessary preprocessing, though it deviates somewhat from the procedures of Torrisi et al.\
`sbfc-hb.py` performs seed-based functional connectivity (SBFC) of preprocessed data for group-level analysis.\
`group-level.py` performs nonlinear transformations on each subject's SBFC _Z_-map to put it in MNI space, then concatenates subject-level SBFC maps, and runs a group-level test using FSL's Randomise, with 5000 permutations and threshold-free cluster enhancement.
## 2. Replication in Human Connectome Project 7T data
`sbfc-hb-hcp.py` is analogous to `sbfc-hb.py`, but performs per-run SBFC per subject and then collapses across runs.\
`group-level.py` is used similarly here, but without nonlinear transformation to MNI space as preprocessed HCP data is alread in MNI space.
## 3. Extension to understand individual differences in habenula connectivity with respect to a myriad of depression-, reward-, and mood-related behavioral variables.
`hcp-indiv-diff-hcp.py` assesses individual differences using a permuted OLS anaysis, controlling for age, gender, and handedness. Each variable in the `model` variables are treated as independent (which, admittedly, may not be the case). Significance threshold for related functional conectivity within each model is corrected by an adapted Sidak procedure introduced by Li & Ji (2005).\
`decoding_maps.py` leverages prior literature indexed in the Neurosynth database to perform meta-analytic functional decoding for a relatively unbiased look at the terms frequently used across manuscripts that report similar activation patterns to our significant results from the HCP dataset.
