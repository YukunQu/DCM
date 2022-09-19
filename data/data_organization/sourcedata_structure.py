import os

def sourcedata_structure(sourcedata_dir,sub_id):
    sub_dir = os.path.join(sourcedata_dir,sub_id)
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir,'Behaviour'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','train_dim1'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','train_dim2'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','train_recall_run1'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','train_recall_run2'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','mixed_test'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','meg_task-1DInfer'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','pilot'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game1'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game2-train'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game2-test'))
    os.mkdir(os.path.join(sub_dir,'Behaviour','placement'))
    os.mkdir(os.path.join(sub_dir,'NeuroData'))
    os.mkdir(os.path.join(sub_dir,'NeuroData','MRI'))
    os.mkdir(os.path.join(sub_dir,'NeuroData','MEG'))
    print(sub_id,'is finished.')


if __name__ == '__main__':
    for i in range(168,201):
        sourcedata_dir = '/mnt/data/Sourcedata/DCM'
        sub_id = 'sub_' + str(i).zfill(3)
        sourcedata_structure(sourcedata_dir,sub_id)