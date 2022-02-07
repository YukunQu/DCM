# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 18:11:27 2021

@author: QYK
"""
import os
import shutil

#%%
def mv_dir(ori_dir,target_dir):
    filename_list = os.listdir(ori_dir)
    for file in filename_list:
        file_oripath = os.path.join(ori_dir,file)
        file_tarpath = os.path.join(target_dir,file)
        shutil.move(file_oripath, file_tarpath)
        
ori_dir =  r'/media/dell/EDB7-93A4/20211205_K207_SUB015'
target_dir =  r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_015/NeuroData/MRI'
mv_dir(ori_dir,target_dir)

#%%
# sort data
def sort_data(origin_dir, target_dir):
    files_name = os.listdir(origin_dir)
    for file in files_name:
        dcmfiles = os.listdir(os.path.join(origin_dir,file))
        for dcmfile in dcmfiles:
            dcmfile_path = os.path.join(origin_dir,file,dcmfile)
            target_path = os.path.join(target_dir,dcmfile)
            shutil.copy(dcmfile_path,target_path)

#%%
def rename_img(imgdir):
	#rename all the images 
	img_list = os.listdir(imgdir)
	for i,img_name in enumerate(img_list):
	    img_ori_name = os.path.join(imgdir,img_name)
	    img_new_name = os.path.join(imgdir,'{}.png'.format(i+1))
	    os.rename(img_ori_name,img_new_name)

#%%
def sourcedata_structure(sourcedata_dir):
    for i in range(1,101):
        sub_id = 'sub_' + str(i).zfill(3)
        sub_dir = os.path.join(sourcedata_dir,sub_id)
        os.mkdir(sub_dir)
        os.mkdir(os.path.join(sub_dir,'Behaviour'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','train_dim1'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','train_dim2'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','train_recall_run1'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','train_recall_run2'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','meg_task-1DInfer'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','pilot'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game1'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game2-train'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','fmri_task-game2-test'))
        os.mkdir(os.path.join(sub_dir,'Behaviour','placement'))
        os.mkdir(os.path.join(sub_dir,'NeuroData'))
        os.mkdir(os.path.join(sub_dir,'NeuroData','MRI'))
        os.mkdir(os.path.join(sub_dir,'NeuroData','MEG'))

#%%
data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata'
subjects = os.listdir(data_dir)
subjects.sort()
for sub in subjects:
    sub_beh_dir = os.path.join(data_dir, sub,'Behaviour','total_test')
    os.mkdir(sub_beh_dir)
    
#%%
def sourcedata_structure(sourcedata_dir):
    for i in range(1,101):
        sub_id = 'sub_' + str(i).zfill(3)
        sub_dir = os.path.join(sourcedata_dir,sub_id)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        os.mkdir(os.path.join(sub_dir,'train_dim1'))
        os.mkdir(os.path.join(sub_dir,'train_dim2'))
        os.mkdir(os.path.join(sub_dir,'train_recall_run1'))
        os.mkdir(os.path.join(sub_dir,'train_recall_run2'))
        os.mkdir(os.path.join(sub_dir,'meg_task-1DInfer'))
        os.mkdir(os.path.join(sub_dir,'pilot'))
        os.mkdir(os.path.join(sub_dir,'fmri_task-game1'))
        os.mkdir(os.path.join(sub_dir,'fmri_task-game2-train'))
        os.mkdir(os.path.join(sub_dir,'fmri_task-game2-test'))

sourcedata_dir = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour'
sourcedata_structure(sourcedata_dir)        