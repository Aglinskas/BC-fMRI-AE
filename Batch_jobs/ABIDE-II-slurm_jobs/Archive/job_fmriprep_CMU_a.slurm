#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEI-CMU_a
#SBATCH --output=./fmriprep_ABIDEI-CMU_a/output_fmriprep_ABIDEI-CMU_a_%a
#SBATCH --error=./fmriprep_ABIDEI-CMU_a/error_fmriprep_ABIDEI-CMU_a_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-14


module load singularity/

date

#rm -rf ./fmriprep_ABIDEI-CMU_a/output_*
#rm -rf ./fmriprep_ABIDEI-CMU_a/error_*

cd $HOME

data_dir=$HOME'/data/ABIDE/RawDataBIDS/CMU_a/'
sing_image=$HOME/fmriprep.simg

cd $data_dir
files=(sub*)

cd $HOME

output_dir=$HOME/data/ABIDE/Derivatives/fmriprep2/

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

#echo ${files[$SLURM_ARRAY_TASK_ID]}

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --output-spaces MNI152NLin2009cAsym:res-2




