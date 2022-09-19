import pandas as pd

#%%
# split the mix_offline performance into the 2 columns: train_ap; train_dp
data = pd.read_csv(r'/mnt/workdir/DCM/tmp/participants.tsv',sep='\t')
data_part1 = data[:145]
data_part2 = data[145:]
mix_offline_acc = data_part2['mix_offline']

train_ap = []
train_dp = []
i = 147
for acc in mix_offline_acc:
    print(i)
    ap_acc,dp_acc = acc.split('/')
    train_ap.append(float(ap_acc))
    train_dp.append(float(dp_acc))
    i+=1

data_part2['train_ap'] = train_ap
data_part2['train_dp'] = train_dp

new_data = pd.concat([data_part1,data_part2])

new_data.to_csv(r'/mnt/workdir/DCM/tmp/participants_new.tsv',sep='\t')

#%%

