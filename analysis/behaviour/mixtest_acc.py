import os.path
import pandas as pd
#%%
# read training accuracy from  mixed test files
sourcedata_dir = r'/mnt/workdir/DCM/sourcedata'
sub_list = [f'sub_{str(s).zfill(3)}' for s in range(203,206)]
sub_list.reverse()
sub_mix_acc = pd.DataFrame({})
for sub in sub_list:
    print(sub)
    mix_test_dir = os.path.join(sourcedata_dir,sub,'Behaviour','mixed_test')
    try:
        mix_test_files = os.listdir(mix_test_dir)
        for f in mix_test_files:
            if '.csv' in f:
                try:
                    tmp_df = pd.read_csv(os.path.join(mix_test_dir,f))
                    id = tmp_df['participant'][0]
                    name = tmp_df['姓名'][0]
                    ap_acc = tmp_df['ap_acc'].max()
                    dp_acc = tmp_df['dp_acc'].max()
                    time = tmp_df['date'][0]
                    sub_mix_acc = sub_mix_acc.append({'sub_id':sub,'exp_id':id,
                                                      'name':name,'AP':ap_acc,'DP':dp_acc,
                                                      'time':time},ignore_index=True)
                except:
                    print(f"The file {f} of {sub} have bugs.")
    except:
        print(f"The {sub} doesn't have the directory.")
#%%
# check data for mixed test performance
beha_total_score = r'/mnt/workdir/DCM/BIDS/participants.tsv'
data = pd.read_csv(beha_total_score,sep='\t')

equal_state = pd.DataFrame()
for index,sub_acc in sub_mix_acc.iterrows():
    sub_ap_acc = sub_acc['AP']
    sub_dp_acc = sub_acc['DP']
    sub_id = sub_acc['sub_id'].replace("_",'-')
    sub_record_ap_acc = data[data['Participant_ID']==sub_id]['train_ap'].values[0]
    sub_record_dp_acc = data[data['Participant_ID']==sub_id]['train_dp'].values[0]
    age = data[data['Participant_ID']==sub_id]['Age'].values[0]
    time = sub_acc['time']
    if (sub_ap_acc==sub_record_ap_acc) and (sub_dp_acc==sub_record_dp_acc):
        equal_state = equal_state.append({'sub_id':sub_id,'state':'equal',
                                          'data_ap':sub_ap_acc,'data_dp':sub_dp_acc,
                                          'doc_ap':sub_record_ap_acc,'doc_dp':sub_record_dp_acc,
                                          'Age':age,'time':time},
                                         ignore_index=True)
    else:
        if sub_id in equal_state['sub_id'].to_list():
            continue
        else:
            equal_state = equal_state.append({'sub_id':sub_id,'state':'not equal',
                                              'data_ap':sub_ap_acc,'data_dp':sub_dp_acc,
                                              'doc_ap':sub_record_ap_acc,'doc_dp':sub_record_dp_acc,
                                              'Age':age,'time':time},
                                             ignore_index=True)


#%%
# split the mix_offline performance into the 2 columns: train_ap; train_dp
data = pd.read_csv(r'/mnt/workdir/DCM/tmp/participants.tsv',sep='\t')
data_part1 = data[:76]
data_part2 = data[76:]
mix_offline_acc = data_part2['mix_offline']
#%%
train_ap = []
train_dp = []
i = 76
for acc in mix_offline_acc:
    print(i)
    ap_acc,dp_acc = acc.split('/')
    train_ap.append(float(ap_acc))
    train_dp.append(float(dp_acc))
    i+=1

data_part2['train_ap'] = train_ap
data_part2['train_dp'] = train_dp

new_data = pd.concat([data_part1,data_part2])
new_data.to_csv(r'/mnt/workdir/DCM/tmp/participants.tsv',sep='\t')