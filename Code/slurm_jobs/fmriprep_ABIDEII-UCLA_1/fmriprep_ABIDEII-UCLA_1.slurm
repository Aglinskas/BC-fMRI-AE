#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEII-UCLA_1
#SBATCH --output=output_fmriprep_ABIDEII-UCLA_1_%a
#SBATCH --error=error_fmriprep_ABIDEII-UCLA_1_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-32

#rm -rf fmriprep_ABIDEII-UCLA_1.slurm;nano fmriprep_ABIDEII-UCLA_1.slurm
#sbatch fmriprep_ABIDEII-UCLA_1.slurm
#$SLURM_ARRAY_TASK_ID

module load singularity/

this_dir=$(pwd);echo $this_dir

data_dir=$HOME'/data/ABIDE2/RawData/ABIDEII-UCLA_1/'

sing_image=$HOME/fmriprep.simg

cd $data_dir
files=(sub*)
cd $this_dir

output_dir=$HOME'/data/ABIDE2/Derivatives/test/'

fs_lic=$HOME'/fs_licence.txt'

echo "${#files[@]} files found in ${data_dir}"

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

#echo ${files[$SLURM_ARRAY_TASK_ID]}

singularity run --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic











