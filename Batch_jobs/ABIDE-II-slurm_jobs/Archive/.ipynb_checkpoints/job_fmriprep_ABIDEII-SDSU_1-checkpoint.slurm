#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEII-SDSU_1
#SBATCH --output=./fmriprep_ABIDEII-SDSU_1/output_fmriprep_ABIDEII-SDSU_1_%a
#SBATCH --error=./fmriprep_ABIDEII-SDSU_1/error_fmriprep_ABIDEII-SDSU_1_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-58

#rm -rf fmriprep_ABIDEII-SDSU_1.slurm;nano fmriprep_ABIDEII-SDSU_1.slurm
#sbatch fmriprep_ABIDEII-SDSU_1.slurm
#$SLURM_ARRAY_TASK_ID


#rm -rf output_fmriprep_ABIDEII-SDSU_1_%a
#rm -rf error_fmriprep_ABIDEII-SDSU_1_%a

module load singularity/

#this_dir=$(pwd);echo $this_dir
cd $HOME

data_dir=$HOME'/data/ABIDE2/RawData/ABIDEII-SDSU_1/'

sing_image=$HOME/fmriprep.simg

cd $data_dir
files=(sub*)
#cd $this_dir

cd $HOME

output_dir=$HOME'/data/ABIDE2/Derivatives/test/'

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

#echo ${files[$SLURM_ARRAY_TASK_ID]}

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming

