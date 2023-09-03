#!/bin/bash

participants_tsv="/mnt/workdir/DCM/BIDS/participants.tsv"
output_dir="/mnt/workdir/DCM/BIDS/derivatives/freesurfer"

# 提取 subject_list
subject_list=$(awk -F'\t' '$3 >= 0.5 {print $1}' $participants_tsv | grep -v "sub-249")

# 定义 recon-all 命令
reconall_cmd() {
    sub_id="$1"
    mri_template="/mnt/workdir/DCM/BIDS/${sub_id}/anat/${sub_id}_T1w.nii.gz"
    cmd="recon-all -s ${sub_id} -i ${mri_template} -all -sd ${output_dir}"
    echo "${cmd}"
    eval "${cmd}"
    echo "${sub_id} finished!"
}

export -f reconall_cmd

# 使用 GNU Parallel 运行 recon-all
starttime=$(date +%s)
echo "subject_list: ${subject_list}"
parallel -j 70 reconall_cmd ::: ${subject_list}
endtime=$(date +%s)

# 计算总时间
total_time=$(( (endtime - starttime) / 3600 ))
echo "总共的时间为: ${total_time} h"

