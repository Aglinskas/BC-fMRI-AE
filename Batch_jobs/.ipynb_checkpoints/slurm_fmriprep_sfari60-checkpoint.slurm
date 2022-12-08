#!/bin/bash

#SBATCH --job-name=job_fmriprep_sfari60
#SBATCH --output=./sfari50-fmriprep/output_sfari60-fmriprep_%a
#SBATCH --error=./sfari50-fmriprep/error_sfari60-fmriprep_%a
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb
#SBATCH --array=0-122

module load singularity/

date

#echo $HOME

cd $HOME

data_dir=$HOME'/data/sfari_bids/ds-sfari-60orfix'
sing_image=$HOME/fmriprep-22

cd $data_dir
files=(sub*)

cd $HOME

output_dir=$HOME/data/sfari_bids/fmriprep/

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub


if [ -d "$output_dir$sub" ];
then
    echo 'already exists'
    echo $output_dir$sub
else
    singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming --output-spaces MNI152NLin2009cAsym:res-2 --return-all-components
fi

