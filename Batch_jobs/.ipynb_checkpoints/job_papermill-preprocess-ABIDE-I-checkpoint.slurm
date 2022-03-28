#!/bin/bash
#SBATCH --job-name=papermill-preprocess-ABIDE-I
#SBATCH --output=ABIDE-I-preproc-jobs/output-preprocess-ABIDE-I_%a.txt
#SBATCH --error=ABIDE-I-preproc-jobs/error-preprocess-ABIDE-I_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --array=0-1200
##SBATCH --mail-user=aglinska@bc.edu
##SBATCH --mail-type=ALL

#rm -rf job_papermill-preprocess-ABIDE-I.slurm; nano job_papermill-preprocess-ABIDE-I.slurm
#sbatch job_papermill-preprocess-ABIDE-I.slurm

bash

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID

notebook_name=~/BC-fMRI-AE/Notebooks/00-ABIDE-I-compcorr-filter.ipynb
outname=~/BC-fMRI-AE/Code/ABIDE-I-preproc-notebooks/preproc-$subID.ipynb

echo $subID
echo $notebook_name
echo $outname

papermill $notebook_name $outname -p s $subID --autosave-cell-every 5 --inject-input-path --progress-bar










