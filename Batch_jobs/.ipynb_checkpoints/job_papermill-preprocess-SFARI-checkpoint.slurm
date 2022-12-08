#!/bin/bash
#SBATCH --job-name=papermill-preprocess-SFARI-I
#SBATCH --output=papermill-SFARI-preproc/output-papermill-preprocess-SFARI-I_%a.txt
#SBATCH --error=papermill-SFARI-preproc/error-papermill-preprocess-SFARI-I_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --array=0-202

bash

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID


notebook_name=~/BC-fMRI-AE/Notebooks/004-SFARI-compcorr-filter.ipynb
outname=~/BC-fMRI-AE/Assets/data_preprocessed_sfari-run1_notebooks/$subID-run2-sfari-compcorr.ipynb

echo $subID
echo $notebook_name
echo $outname

papermill $notebook_name $outname -p s $subID --autosave-cell-every 5 --inject-input-path --progress-bar










