import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr
from nilearn.image import index_img, mean_img, load_img, resample_to_img
from nilearn.masking import apply_mask


def cal_map_rdm(condition):
    x = range(1,6)
    y = range(1,6)
    d1,d2 = np.meshgrid(x,y)
    d1 = d1.reshape(-1)
    d2 = d2.reshape(-1)

    map = np.zeros((25,2))
    for i, (x,y) in enumerate(zip(d1,d2)):
        map[i][0] = x
        map[i][1] = y
    map = map[1:-1]
    obs_des = {'conds': condition}
    map = rsd.Dataset(measurements=map,obs_descriptors=obs_des)
    model_RDM = rsr.calc_rdm(map,method='euclidean',descriptor='conds')
    model_RDM = rsatoolbox.rdm.sqrt_transform(model_RDM)
    return model_RDM


def rsa(subjects,condition,roi_name):
    fmri_measurement = []
    for i,sub_id in enumerate(subjects):
        cons_img_path = []
        for j,con_name in enumerate(condition):
            con_img_path = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/Setall/6fold/' \
                           r'sub-{}/con_{}.nii'.format(sub_id,str(j+1).zfill(4))
            cons_img_path.append(con_img_path)

        roi_path = roi_template.format(roi_name)
        cons_roi_data = apply_mask(cons_img_path,roi_path)

        des = {'session': 'game1', 'subj': sub_id}
        obs_des = {'conds': np.array(condition)}
        nVox = cons_roi_data.shape[-1]
        chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
        fmri_measurement.append(rsd.Dataset(measurements=cons_roi_data,
                                            descriptors=des,
                                            obs_descriptors=obs_des,
                                            channel_descriptors=chn_des))

    # Calculating the RDM for brain activity of different conditions
    sub_rdms = rsr.calc_rdm(fmri_measurement,descriptor='conds', method='mahalanobis')
    sub_rdms =  rsatoolbox.rdm.sqrt_transform(sub_rdms)

    # create RDM for hypothetical 2D cognitive map
    model_RDM = cal_map_rdm(condition)

    subs_sim = []
    for sub_rdm in sub_rdms:
        sim = rsatoolbox.rdm.compare(sub_rdm,model_RDM,method='tau-a')
        subs_sim.extend(sim[0])
    return sub_rdms, model_RDM, subs_sim

# Defining the data set
# specify subjects
participants_tsv = r'/mnt/workdir/DCM/BIDS/participants.tsv'
participants_data = pd.read_csv(participants_tsv, sep='\t')
data = participants_data.query('game1_fmri==1')

pid = data['Participant_ID'].to_list()
subjects = [p.split('_')[-1] for p in pid]

rois = ['HCl','HCr','ECl','ECr','EC_Grid','M1l','M1r']
roi_template = r'/mnt/workdir/DCM/docs/Reference/Park_Grid_ROI/{}_roi.nii'

condition = ['pos'+str(i).zfill(2) for i in range(2, 25)]

roi_label = []
roi_sim = []
for roi_name in rois:
    sub_rdms, model_RDM, subs_sim = rsa(subjects,condition,roi_name)
    roi_label.extend([roi_name]*len(subs_sim))
    roi_sim.extend(subs_sim)
    sub_rdms.save(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/rsa_result/{}_rdms.hdf5'.format(roi_name))
    np.save(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/rsa_result/{}_rs.npy'.format(roi_name),np.array(subs_sim))

model_RDM.save(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/rsa_result/model_RDM.hdf5')
sns.set_theme(palette="pastel")
sns.boxplot(x=roi_label, y=roi_sim,palette=['b'])

#%%
from scipy.stats import pearsonr
acc = data['game1_acc']
age = data['Age']

for roi_name in rois:
    rs = np.load(r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/rsa_result/{}_rs.npy'.format(roi_name))
    res = pearsonr(age,rs)
    r = res[0]
    p = res[1]
    print(roi_name,r,p)

    # plot
    g = sns.jointplot(x=age, y=rs,kind="reg", truncate=False)
    # move overall title up
    g.set_axis_labels('Age', '{}_similarity'.format(roi_name),fontsize=18)
    g.fig.subplots_adjust(top=0.92)
    if p < 0.001:
        g.fig.suptitle('r:{}  p<0.001'.format(round(r,3)))
    else:
        g.fig.suptitle('r:{}, p:{}'.format(round(r,3),round(p,3)),fontsize=18)

    savepath = r'/mnt/workdir/DCM/BIDS/derivatives/Nipype/game1/rsa/rsa_result/plot/age/{}_rs_correlation.png'.format(roi_name)
    g.savefig(savepath,bbox_inches='tight',pad_inches=0,dpi=300)
