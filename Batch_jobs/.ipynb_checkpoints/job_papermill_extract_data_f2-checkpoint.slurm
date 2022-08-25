#!/bin/bash
#SBATCH --job-name=job_papermill_extract_data_f2
#SBATCH --output=./Papermill_extract_data_f2/02_output_%a.txt
#SBATCH --error=./Papermill_extract_data_f2/03_error_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=8gb
#SBATCH --array=0-2117%25
#SBATCH --partition=partial_nodes

date

cd ~/BC-fMRI-AE/Notebooks/

subID=$SLURM_ARRAY_TASK_ID

notebook_name=~/BC-fMRI-AE/Notebooks/004-ABIDE-I-and-II-compcorr-filter.ipynb
outname=~/BC-fMRI-AE/Assets/data_preprocessed_f3_notebooks/03_preprocess_$subID.ipynb

echo $subID
echo $notebook_name
echo $outname

papermill $notebook_name $outname -p s $subID --autosave-cell-every 5 --inject-input-path --progress-bar










