#!/bin/bash
#SBATCH --job-name=papermill-preprocess-SFARI-II
#SBATCH --output=papermill-SFARI-preproc/output-papermill-preprocess-SFARI-II_%a.txt
#SBATCH --error=papermill-SFARI-preproc/error-papermill-preprocess-SFARI-II_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --array=0-205
##SBATCH --mail-user=aglinska@bc.edu
##SBATCH --mail-type=ALL

#rm -rf job_papermill-preprocess-ABIDE-I.slurm; nano job_papermill-preprocess-ABIDE-I.slurm
#sbatch job_papermill-preprocess-ABIDE-I.slurm

bash

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID

notebook_name=~/BC-fMRI-AE/Notebooks/014-SFARI-compcorr-filter.ipynb
outname=~/BC-fMRI-AE/Assets/papermill-SFARI-preproc/$subID-run2-sfari-compcorr.ipynb

echo $subID
echo $notebook_name
echo $outname

papermill $notebook_name $outname -p s $subID -p r 2 --autosave-cell-every 5 --inject-input-path --progress-bar










