import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from os.path import join, exists
from nilearn.mass_univariate import permuted_ols
from nilearn.input_data import NiftiMasker

def jili_sidak_mc(data, alpha):
    import math
    import numpy as np

    mc_corrmat = data.corr()
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('Number of effective comparisons: {0}'.format(M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

subjects = [100610, 102311, 102816, 104416, 105923, 108323, 109123,
            111514, 114823, 115017, 115825, 118225, 125525, 126426, 126931,
            128935, 130114, 130518, 131217, 131722, 132118, 134627, 134829,
            135124, 137128, 140117, 144226, 145834, 146129, 146432, 146735,
            146937, 148133, 155938, 156334, 157336, 158035, 158136, 159239,
            162935, 164131, 164636, 167036, 169343, 995174, 971160, 966975,
            958976, 951457, 943862, 942658, 927359, 926862, 910241, 905147,
            901442, 901139, 899885, 898176, 878877, 878776, 872764, 871762,
            861456, 814649, 789373, 783462, 782561, 771354, 770352, 765864,
            757764, 751550, 601127, 617748, 627549, 638049, 644246, 654552,
            671855, 680957, 690152, 706040, 724446, 725751, 732243, 745555,
            581450, 573249, 572045, 562345, 550439, 547046, 541943, 525541, 467351]

sink_dir = '/Users/Katie/Dropbox/Projects/habenula'
data_dir = '/Users/Katie/Dropbox/Projects/habenula'

unr_data = pd.read_csv(join(data_dir, 'unrestricted_kbott006_11_16_2018_13_50_26.csv'), header=0, index_col=0)
unr_data = unr_data.replace({'M':0, 'F':1})
#unr_data = unr_data.replace({'26-30': 28., '31-35': 33., '22-25': 23.5, '36+':36})
unr_data = unr_data.loc[subjects]

res_data = pd.read_csv(join(data_dir, 'RESTRICTED_kbott006_3_4_2019_12_10_43.csv'), header=0, index_col=0)
res_data = res_data.replace({'M':0, 'F':1})
res_data = res_data.replace({'SSAGA_Depressive_Ep':{1:0}})
res_data = res_data.replace({'SSAGA_Depressive_Ep':{5:1}})
res_data = res_data.loc[subjects]

data = pd.concat([unr_data, res_data], axis=1)

personality_vars = ['NEOFAC_E', 'NEOFAC_C','NEOFAC_N', 'NEOFAC_O', 'NEOFAC_A']
helpless_vars = ['LifeSatisf_Unadj', 'MeanPurp_Unadj',
                         'PosAffect_Unadj', 'PercStress_Unadj', 'SelfEff_Unadj']
social_vars = ['EmotSupp_Unadj', 'InstruSupp_Unadj',
                       'Friendship_Unadj', 'Loneliness_Unadj', 'PercHostil_Unadj',
                       'PercReject_Unadj']
emo_recog_vars = ['ER40ANG', 'ER40FEAR', 'ER40HAP', 'ER40NOE', 'ER40SAD']
smoking = ['SSAGA_FTND_Score', 'Total_Cigarettes_7days', 'SSAGA_TB_DSM_Difficulty_Quitting', 'SSAGA_TB_DSM_Withdrawal']
demo_vars = ['Gender', 'Age_in_Yrs', 'Handedness']
delay_discount = ['DDisc_SV_1mo_200', 'DDisc_SV_6mo_200', 'DDisc_SV_1yr_200', 'DDisc_SV_3yr_200', 'DDisc_SV_5yr_200', 'DDisc_SV_10yr_200', 'DDisc_SV_1mo_40K', 'DDisc_SV_6mo_40K', 'DDisc_SV_1yr_40K', 'DDisc_SV_3yr_40K', 'DDisc_SV_5yr_40K', 'DDisc_SV_10yr_40K']
reward = ['Gambling_Task_Reward_Median_RT_Larger', 'Gambling_Task_Reward_Perc_Smaller', 'Gambling_Task_Reward_Median_RT_Smaller', 'Gambling_Task_Reward_Perc_NLR', 'Gambling_Task_Punish_Perc_Larger', 'Gambling_Task_Punish_Median_RT_Larger', 'Gambling_Task_Punish_Perc_Smaller', 'Gambling_Task_Punish_Median_RT_Smaller']
negative_affect = ['AngAffect_Unadj', 'AngHostil_Unadj', 'AngAggr_Unadj', 'FearAffect_Unadj', 'FearSomat_Unadj', 'Sadness_Unadj']
depression = ['FamHist_Moth_Dep', 'FamHist_Fath_Dep', 'DSM_Depr_Raw', 'SSAGA_Depressive_Ep', 'SSAGA_Depressive_Sx']
models = [delay_discount, reward]

sns.distplot(data[personality_vars[4]], hist=False, rug=True)
for model in models:
    print(model)
    p_cor = jili_sidak_mc(data[model], 0.05)
    test = permuted_ols(data[model].values, fmri_masked, confounding_vars=data[demo_vars], model_intercept=True)
    for i in np.arange(0,len(test[0])):
        if np.max(test[0][i]) > 1.9:
            unmasked = nifti_masker.inverse_transform(np.ravel(test[0][i]))
            unmasked.to_filename(join(sink_dir, 'osl_nlogpval_hbfc_{0}-alpha_{1}.nii.gz'.format(model[i], 10**-(np.max(test[0][i])))))
            unmasked = nifti_masker.inverse_transform(np.ravel(test[1][i]))
            unmasked.to_filename(join(sink_dir, 'osl_tscores_hbfc_{0}.nii.gz'.format(model[i])))
        else:
            unmasked = nifti_masker.inverse_transform(np.ravel(test[1][i]))
            unmasked.to_filename(join(sink_dir, 'osl_tscores_hbfc_{0}-p>0.05(corr).nii.gz'.format(model[i])))

smoker_data = data[np.isnan(data['SSAGA_TB_DSM_Difficulty_Quitting']) == False]
smokers = smoker_data.index.values
