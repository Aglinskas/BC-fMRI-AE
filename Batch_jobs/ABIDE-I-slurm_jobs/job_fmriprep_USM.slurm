#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEI-USM
#SBATCH --output=./fmriprep_ABIDEI-USM/output_fmriprep_ABIDEI-USM_%a
#SBATCH --error=./fmriprep_ABIDEI-USM/error_fmriprep_ABIDEI-USM_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-101


module load singularity/

date

#rm -rf ./fmriprep_ABIDEI-USM/output_*
#rm -rf ./fmriprep_ABIDEI-USM/error_*

cd $HOME

data_dir=$HOME'/data/ABIDE/RawDataBIDS/USM/'
sing_image=$HOME/fmriprep.simg

cd $data_dir
files=(sub*)

cd $HOME

output_dir=$HOME/data/ABIDE/Derivatives/fmriprep/

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming


