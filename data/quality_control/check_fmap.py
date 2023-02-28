import os
import pandas as pd

#%%
# check how many fmap does subject have?
bids_dir = r'/mnt/workdir/DCM/BIDS'
subjects_list = os.listdir(bids_dir)
subjects_list = [s for s in subjects_list if 'sub-' in s]
subjects_list.sort()

i = 0
sub_name = []
fmaps_num = []
for sub in subjects_list:
    sub_fmap_dir = os.path.join(bids_dir,sub,'fmap')
    if not os.path.exists(sub_fmap_dir):
        i += 1
        # print(sub,f'have none fmap file.')
        sub_name.append(sub)
        fmaps_num.append(0)
    else:
        fmap_list = os.listdir(sub_fmap_dir)
        fmap_num = len(fmap_list)/6
        if fmap_num < 1:
            i += 1
            print(sub,f'have {fmap_num} fmap files.')
            sub_name.append(sub)
            fmaps_num.append(fmap_num)
        else:
            print(sub,f'have {fmap_num} fmap files.')

lack_df = pd.DataFrame({'sub':sub_name, 'fmaps_num':fmaps_num})
#%%
lack_df.to_csv(r"/mnt/workdir/DCM/tmp/lack_df.csv",index=False)
