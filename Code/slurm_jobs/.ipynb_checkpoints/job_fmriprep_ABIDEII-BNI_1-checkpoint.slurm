#!/bin/bash

#SBATCH --job-name=job_fmriprep_{folder}
#SBATCH --output=./fmriprep_{folder}/output_fmriprep_{folder}_%a
#SBATCH --error=./fmriprep_{folder}/error_fmriprep_{folder}_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-{n}

#rm -rf fmriprep_{folder}.slurm;nano fmriprep_{folder}.slurm
#sbatch fmriprep_{folder}.slurm
#$SLURM_ARRAY_TASK_ID


#rm -rf output_fmriprep_{folder}_%a
#rm -rf error_fmriprep_{folder}_%a

module load singularity/

#this_dir=$(pwd);echo $this_dir
cd $HOME

data_dir=$HOME'/data/ABIDE2/RawData/{folder}/'

sing_image=$HOME/fmriprep.simg

cd $data_dir
files=(sub*)
#cd $this_dir

cd $HOME

output_dir=$HOME'/data/ABIDE2/Derivatives/test/'

fs_lic=$HOME'/fs_licence.txt'

sub=${{files[$SLURM_ARRAY_TASK_ID]}} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

#echo ${{files[$SLURM_ARRAY_TASK_ID]}}

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic

