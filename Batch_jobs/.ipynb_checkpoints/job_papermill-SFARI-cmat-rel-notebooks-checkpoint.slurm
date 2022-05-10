#!/bin/bash
#SBATCH --job-name=papermill-cmat-rel-notebooks-SFARI
#SBATCH --output=SFARI-I-cmat-rel-jobs/output-preprocess-ABIDE-I_%a.txt
#SBATCH --error=SFARI-I-cmat-rel-jobs/error-preprocess-ABIDE-I_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --array=0-205
##SBATCH --mail-user=aglinska@bc.edu
##SBATCH --mail-type=ALL

#rm -rf job_papermill-cmat-rel-notebooks.slurm; nano job_papermill-cmat-rel-notebooks.slurm
#sbatch job_papermill-cmat-rel-notebooks.slurm

bash

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID

notebook_name=~/BC-fMRI-AE/Notebooks/016-get-split-cmats-SFARI.ipynb
outname=~/BC-fMRI-AE/Assets/papermill-SFARI-extract/preproc-$subID-run2.ipynb

echo $subID
echo $notebook_name
echo $outname

papermill $notebook_name $outname -p s $subID -p r 2 --autosave-cell-every 5 --inject-input-path --progress-bar










