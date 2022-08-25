#!/bin/bash

#SBATCH --job-name=job_fmriprep_ABIDEI-MaxMun_c
#SBATCH --output=./fmriprep_ABIDEI-MaxMun_c/output_fmriprep_ABIDEI-MaxMun_c_%a
#SBATCH --error=./fmriprep_ABIDEI-MaxMun_c/error_fmriprep_ABIDEI-MaxMun_c_%a

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32gb

#SBATCH --array=0-17


module load singularity/

date

#rm -rf ./fmriprep_ABIDEI-MaxMun_c/output_*
#rm -rf ./fmriprep_ABIDEI-MaxMun_c/error_*

cd $HOME

data_dir=$HOME'/data/ABIDE/RawDataBIDS/MaxMun_c/'
sing_image=$HOME/fmriprep-22

cd $data_dir
files=(sub*)

cd $HOME

output_dir=$HOME/data/ABIDE/Derivatives/fmriprep4/

fs_lic=$HOME'/fs_licence.txt'

sub=${files[$SLURM_ARRAY_TASK_ID]} #;echo $sub

echo $HOME
echo $data_dir
echo $output_dir
echo $sub

singularity run --bind /data/aglinska/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming --output-spaces MNI152NLin2009cAsym:res-2 --return-all-components


