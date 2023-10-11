import os
import pandas as pd


def read_sM(subjects_list,smeasure='Total cortical gray matter volume'):
    fs_dir = "/mnt/workdir/DCM/BIDS/derivatives/freesurfer"
    measures = []
    for sid in subjects_list:
        # Define the path to the aseg.stats file
        stats_file = os.path.join(fs_dir, sid, "stats", "aseg.stats")

        # Open the stats file and read its contents
        with open(stats_file, "r") as f:
            stats_data = f.readlines()

        # Find the line that contains the Total cortical gray matter volume information
        for line in stats_data:
            if smeasure in line:
                # Extract the volume value from the line
                measure = float(line.split(",")[-2])
                measures.append(measure)
                break
    return measures
#%%
beha_total_score = r'/mnt/workdir/DCM/BIDS/participants.tsv'
data = pd.read_csv(beha_total_score,sep='\t')
data = data[data['game1_fmri'] >= 0.5]
#%%
sub_id = data['Participant_ID']
Age = data['Age']
sex = data['Sex ']
sex = [s.capitalize() for s in sex]

# load brain chart template
bc_template = pd.read_csv(r'/mnt/workdir/DCM/Docs/Reference/BrainChart/template.csv')
tem_columns = bc_template.columns

bc_data = {}
bc_data['participant'] = sub_id
bc_data['Age'] = Age
bc_data['sex'] = sex


bc_data['study'] = 'DCM_qyk'
bc_data['fs_version'] = pd.NA
bc_data['country'] = 'China'
bc_data['run'] = 1
bc_data['session'] = 1
bc_data['dx'] = 'CN'

# read the brain measures.
bc_data['GMV'] = read_sM(sub_id,'Total cortical gray matter volume')
bc_data['WMV'] = read_sM(sub_id, 'Total cerebral white matter volume')
bc_data['sGMV'] = read_sM(sub_id, 'Subcortical gray matter volume')
bc_data['Ventricles'] = read_sM(sub_id, 'Volume of ventricles and choroid plexus')
bc_data['INDEX.TYPE'] = pd.NA
bc_data['INDEX.OB'] = pd.NA
bc_data = pd.DataFrame(bc_data)
new_columns = bc_data.columns
for c in tem_columns:
    if c not in new_columns:
        print(c,'column was not found.')

previous_file = pd.read_csv("/mnt/workdir/DCM/Docs/Reference/BrainChart/template-205sub.csv")
previous_file = previous_file[['participant','age_days']]
# compare different participant in two files
p1 = previous_file['participant'].to_list()
p2 = bc_data['participant'].to_list()
for p in p1:
    if p not in p2:
        print(p,'was not found in new_data.')
print('---------------------------------------------')
for p in p2:
    if p not in p1:
        print(p,'was not found in previous data.')

merge_df = pd.merge(previous_file, bc_data, on='participant')
tem_columns = tem_columns.to_list()
tem_columns.remove('Unnamed: 0')
merge_df = merge_df[tem_columns]
merge_df.index = range(1,len(merge_df)+1)
merge_df.to_csv('/mnt/workdir/DCM/Docs/Reference/BrainChart/DCM_qyk.csv')