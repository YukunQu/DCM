#!/bin/bash

participants_tsv="/mnt/workdir/DCM/BIDS/participants.tsv"
output_dir="/mnt/workdir/DCM/BIDS/derivatives/freesurfer"

# 提取 subject_list
subject_list=$(awk -F'\t' 'NR==1{for (i=1; i<=NF; i++) if ($i == "game1_fmri") col=i} NR>1 && $col >= 0.5 {print $1}' $participants_tsv | grep -v "sub-249")

echo "Extracted subject list : ${subject_list}"
subject_count=$(echo "$subject_list" | wc -l)
echo "被试数量: $subject_count"

# 定义 recon-all 命令
reconall_cmd() {
    sub_id="$1"
    mri_template="/mnt/workdir/DCM/BIDS/${sub_id}/anat/${sub_id}_T1w.nii.gz"
    cmd="recon-all -s ${sub_id} -i ${mri_template} -all -sd /mnt/workdir/DCM/BIDS/derivatives/freesurfer"
    echo "Running command: ${cmd}"
    eval "${cmd}"
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo "${sub_id} finished successfully!"
    else
        echo "Error: ${sub_id} failed with exit status ${exit_status}"
    fi
}

export -f reconall_cmd

# 定义每批次的最大被试数量
max_subjects_per_batch=70

# 将 subject_list 分割成子列表，并循环运行 recon-all
for ((i=0; i<$subject_count; i+=$max_subjects_per_batch)); do
    echo "Processing subjects $((i+1)) to $((i+max_subjects_per_batch))..."
    chunk=$(echo "$subject_list" | awk "NR > $i && NR <= $((i+max_subjects_per_batch))")
    starttime=$(date +%s)
    echo "subject_list: ${chunk}"
    parallel -j 70 reconall_cmd ::: ${chunk}
    endtime=$(date +%s)

    # 计算总时间
    total_seconds=$((endtime - starttime))
    total_hours=$((total_seconds / 3600))
    total_minutes=$(((total_seconds / 60) % 60))
    total_seconds=$((total_seconds % 60))
    echo "本批次运行时间为: ${total_hours}h ${total_minutes}m ${total_seconds}s"
done


