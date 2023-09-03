import os
import pandas as pd

# get the subject directory list
dti_dir = r'/mnt/data/DCM/HNY/Diffusion/DTI/DTI'
sub_dir_list = os.listdir(dti_dir)

# get the file list
csv_file_names = [f for f in os.listdir(os.path.join(dti_dir,sub_dir_list[0])) if f.endswith(".csv")]

# set save_dir
save_dir = r'/mnt/data/DCM/HNY/Diffusion/DTI/Organized'

# loop the file names, merge each type of file across subjects
for file_name in csv_file_names:
    # create a null dataframe
    merge_df = pd.DataFrame()
    for index,sub in enumerate(sub_dir_list):
        # read the individual subject's csv file
        csv_path = os.path.join(dti_dir, sub, file_name)
        df = pd.read_csv(csv_path)

        # get label(columns)
        if index == 0:
            labels = df['name']

        # get the sub_id from file_name
        sub_id = 'sub-'+sub.split('_')[3]
        metric_name = list(df.columns)
        metric_name.remove("name")
        sub_metric = {}
        for label in labels:
            metric = df.loc[df.name==label,metric_name].values[0][0]
            sub_metric[label] = metric

        sub_metric['Subject'] = sub_id
        merge_df = merge_df.append(sub_metric,ignore_index=True)

    # sort by column 'A' and make it the first column
    merge_df = merge_df.sort_values('Subject').set_index('Subject')
    # save into the save_dir with same name
    save_path = os.path.join(save_dir, 'AllSub_'+file_name)
    merge_df.to_csv(save_path, index=True)

