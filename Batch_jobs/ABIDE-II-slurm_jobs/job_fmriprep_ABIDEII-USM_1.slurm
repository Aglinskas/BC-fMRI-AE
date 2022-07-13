#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEII-USM_1
#SBATCH --output=./fmriprep_ABIDEII-USM_1/output_fmriprep_ABIDEII-USM_1_%a
#SBATCH --error=./fmriprep_ABIDEII-USM_1/error_fmriprep_ABIDEII-USM_1_%a
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb
#SBATCH --array=0-33

module load singularity/

date

#rm -rf ./fmriprep_ABIDEII-USM_1/output_*
#rm -rf ./fmriprep_ABIDEII-USM_1/error_*

cd $HOME

data_dir=$HOME'/data/ABIDE2/RawData/ABIDEII-USM_1/'
sing_image=$HOME/fmriprep-22

cd $data_dir
files=(sub*)

cd $HOME

output_dir=$HOME'/data/ABIDE2/Derivatives/fmriprep2/'

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub


echo $HOME
echo $data_dir
echo $output_dir
echo $sub

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming --output-spaces MNI152NLin2009cAsym:res-2


