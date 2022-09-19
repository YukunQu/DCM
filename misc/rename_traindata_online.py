import os
import pandas as pd


def rename_data(train_data_dir,filename):
    train_data_path = os.path.join(train_data_dir,filename)
    train_data = pd.read_csv(train_data_path)
    participant = train_data['participant'].to_list()[0]
    name = train_data['姓名'].to_list()[0]
    date = train_data['date'].to_list()[0].split('_')[0]

    new_fn = filename.split('_')[0] + f'_{participant}_{name}_{date}.csv'
    savepath = os.path.join(train_data_dir, new_fn)
    train_data.to_csv(savepath,index=False)


if __name__ == '__main__':
    trian_data_dir = r'/mnt/workdir/DCM/docs/被试招募及训练/Online_traindata/脑岛/psychoJs'
    files = os.listdir(trian_data_dir)
    for filename in files:
        rename_data(trian_data_dir,filename)