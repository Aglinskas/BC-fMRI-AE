#!/bin/bash

#SBATCH --job-name=sfari60_fmriprep
#SBATCH --output=output_sfari60_fmriprep_%a
#SBATCH --error=error_sfari60_fmriprep_%a

#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32gb
#SBATCH --array=1-122

#rm -rf job_sfari60_fmriprep.slurm;nano job_sfari60_fmriprep.slurm
#sbatch job_sfari60_fmriprep.slurm
#$SLURM_ARRAY_TASK_ID

module load singularity/

this_dir=$(pwd) # Where we launch from

sing_image=$HOME/fmriprep.simg
fs_lic=$HOME'/fs_licence.txt'

data_dir=$HOME'/data/sfari_bids/ds-sfari-60orfix/'
cd $data_dir
files=(sub*)

cd $this_dir

output_dir=$HOME'/data/sfari_bids/'


echo "${#files[@]} files found in ${data_dir}"

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $this_dir
echo $data_dir
echo $output_dir
echo $sub

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic

